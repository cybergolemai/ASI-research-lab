"""
Replication harness for the baseline filler-token experiments in "Not All LLM
Reasoning is Visible in the Chain-of-Thought" (arXiv:2607.22925).

What this measures
------------------
For each (model, task, filler condition) it runs a no-chain-of-thought prompt
with K few-shot examples and scores exact-match accuracy against the gold
integer. Conditions differ only in the filler sequence prefilled before
"Answer:", so a condition's delta against the same model's `baseline` condition
is the filler-token uplift the paper reports in Table 1 and Figure 1.

Following the paper:
  - all problems within a task are shared across conditions and models, so
    comparisons are paired (Section 3.2);
  - sampling is temperature 1, top-p 0.95;
  - accuracy carries a 95% confidence interval from the binomial standard error;
  - samples that emit explicit reasoning instead of an answer are discarded and
    counted separately (Appendix A.4, selection effects).

Usage
-----
    python -m scripts.filler_eval --config runs/filler_baseline.json
    python -m scripts.filler_eval --config runs/filler_baseline.json --limit 100
    python -m scripts.filler_eval --config runs/filler_baseline.json --dry-run

Output
------
A JSON record and a readable summary are printed to stdout. The same JSON is
written to `$ORX_RUN_DIR/results/` when the run is managed by OpenResearch, and
to `./results/` otherwise.
"""
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass

from nanochat import orx_run as orx
from nanochat.filler_backends import BackendError, ResponseCache, make_backend
from nanochat.filler_prompts import NO_THINK_PREFIX, build_messages
from nanochat.filler_tasks import build_task
from nanochat.filler_tokens import FILLER_TYPES, build_filler, get_token_counter

RESULTS_SCHEMA_VERSION = 1


# -----------------------------------------------------------------------------
# Scoring

_ANSWER_RE = re.compile(r"-?\d[\d,]*")


def parse_answer(text):
    """
    Extract the model's numeric answer, or None.

    Prefilled generation normally yields just the number. When a model echoes
    the cue or wraps the number in prose, the first number is taken; a bare
    "N/A" or an empty completion is not an answer.
    """
    if not text:
        return None
    match = _ANSWER_RE.search(text)
    if not match:
        return None
    try:
        return int(match.group(0).replace(",", ""))
    except ValueError:
        return None


_ANSWER_ANCHOR_RE = re.compile(r"answer\s*:\s*(-?\d[\d,]*)", re.IGNORECASE)
_ANSWER_CUE_RE = re.compile(r"answer\s*:", re.IGNORECASE)


def has_answer_cue(text):
    """
    True when the completion wrote the "Answer:" cue.

    A truncated CoT trace usually still contains numbers, so `parse_answer_cot`
    falls back to the last one and the cell is scored rather than skipped. That
    fallback is a guess, so the cue's presence is tracked separately: for a
    token-capped CoT condition, 100 - explicit_rate is the truncation rate.
    """
    return bool(_ANSWER_CUE_RE.search(text or ""))


def parse_answer_cot(text):
    """
    Extract the final answer from a chain-of-thought completion.

    The CoT condition ends with "Answer: N", so anchor on that cue and take the
    last match. If the model never writes the cue, fall back to the last number
    in the text rather than the first, since the first number is usually part of
    the reasoning.
    """
    if not text:
        return None
    matches = _ANSWER_ANCHOR_RE.findall(text)
    if matches:
        try:
            return int(matches[-1].replace(",", ""))
        except ValueError:
            return None
    numbers = _ANSWER_RE.findall(text)
    if not numbers:
        return None
    try:
        return int(numbers[-1].replace(",", ""))
    except ValueError:
        return None


def looks_like_reasoning(text):
    """
    True when the completion contains an explanation rather than a bare answer.

    The paper discards these samples because they are a selection effect: a
    model that reasons and then answers is solving an easier problem than the
    no-CoT prompt asked for (Appendix A.4).
    """
    if not text:
        return False
    stripped = re.sub(r"^\s*answer\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    match = _ANSWER_RE.search(stripped)
    prefix = stripped[: match.start()] if match else stripped
    words = re.findall(r"[A-Za-z]{2,}", prefix)
    return len(words) >= 2 or len(stripped.split()) > 12


def confidence_interval(accuracy, n):
    """95% CI half-width from the binomial standard error, in percentage points."""
    if not n:
        return 0.0
    return 100.0 * 1.96 * math.sqrt(max(accuracy * (1.0 - accuracy), 0.0) / n)


@dataclass
class Tally:
    n: int = 0
    kept: int = 0
    correct: int = 0
    correct_all: int = 0
    reasoning: int = 0
    answered: int = 0
    explicit: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add(self, text, gold, prompt_tokens=0, completion_tokens=0, cot=False):
        self.n += 1
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        answer = parse_answer_cot(text) if cot else parse_answer(text)
        self.answered += int(answer is not None)
        self.explicit += int(has_answer_cue(text))
        is_correct = answer == gold
        self.correct_all += int(is_correct)
        if not cot and looks_like_reasoning(text):
            self.reasoning += 1
            return
        self.kept += 1
        self.correct += int(is_correct)

    def summary(self):
        accuracy = self.correct / self.kept if self.kept else 0.0
        accuracy_all = self.correct_all / self.n if self.n else 0.0
        return {
            "n": self.n,
            "n_kept": self.kept,
            "n_reasoning_discarded": self.reasoning,
            "n_answered": self.answered,
            "answer_rate": round(100.0 * self.answered / self.n, 2) if self.n else 0.0,
            "n_explicit": self.explicit,
            "explicit_rate": round(100.0 * self.explicit / self.n, 2) if self.n else 0.0,
            "accuracy": round(100.0 * accuracy, 2),
            "ci95": round(confidence_interval(accuracy, self.kept), 2),
            "accuracy_including_discarded": round(100.0 * accuracy_all, 2),
            "reasoning_rate": round(100.0 * self.reasoning / self.n, 2) if self.n else 0.0,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


# -----------------------------------------------------------------------------
# Config

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    for field in ("models", "tasks", "conditions"):
        if field not in config:
            raise ValueError(f"config {path} is missing required field {field!r}")
    unknown = {c["filler_type"] for c in config["conditions"] if c.get("filler_type")} - set(FILLER_TYPES)
    if unknown:
        raise ValueError(f"unknown filler types in config: {sorted(unknown)}")
    return config


def model_setting(config, model, key, default=None):
    """Resolve a per-model setting, supporting exact ids and shell-style prefixes."""
    table = config.get("prompting", {}).get(key) or {}
    if model in table:
        return table[model]
    for pattern, value in table.items():
        if pattern.endswith("*") and model.startswith(pattern[:-1]):
            return value
    return default


def system_prefix(config, model):
    prefix = model_setting(config, model, "system_prefix", "") or ""
    if model_setting(config, model, "no_think", False):
        prefix = f"{NO_THINK_PREFIX}\n{prefix}".strip()
    return prefix


# -----------------------------------------------------------------------------
# Execution

DEFAULT_COT_MAX_TOKENS = 2048


def build_conditions(config, counter):
    """
    Build the condition list, calibrating each filler once.

    A condition may carry its own `max_tokens` (and `reasoning`) to override the
    shared `sampling` block, so one config can hold several chain-of-thought
    conditions at different caps. CoT conditions default to `cot_max_tokens`
    (2048) with reasoning enabled; no-CoT conditions keep the shared sampling
    values unless they say otherwise.
    """
    conditions = []
    default_cot = int(config.get("cot_max_tokens", DEFAULT_COT_MAX_TOKENS))
    for spec in config["conditions"]:
        cap = spec.get("max_tokens")
        if spec.get("cot"):
            conditions.append({
                "name": spec["name"],
                "filler": {"type": None, "text": "", "words": 0, "tokens": 0,
                           "target_tokens": 0, "calibration": None},
                "cot": True,
                "max_tokens": int(cap) if cap is not None else default_cot,
                "reasoning": bool(spec.get("reasoning", True)),
            })
            continue
        kind = spec.get("filler_type")
        tokens = int(spec.get("filler_tokens", 0))
        words = spec.get("filler_words")
        if kind:
            filler = build_filler(kind, tokens, counter, n_words=words)
        else:
            filler = {"type": None, "text": "", "words": 0, "tokens": 0,
                      "target_tokens": 0, "calibration": None}
        conditions.append({
            "name": spec["name"],
            "filler": filler,
            "cot": False,
            "max_tokens": int(cap) if cap is not None else None,
            "reasoning": bool(spec["reasoning"]) if "reasoning" in spec else None,
        })
    return conditions


def run_cell(backend, problems, examples, filler, config, model, limit=None, prefill=True, bar=None, cot=False):
    """Evaluate one (model, task, condition) cell and return a Tally plus samples."""
    prefix = system_prefix(config, model)
    system_prompt = config.get("prompting", {}).get("system_prompt")
    tally = Tally()
    samples = []
    problems = problems[:limit] if limit else problems

    def one(problem):
        messages = build_messages(
            problem, examples, filler_text=filler["text"], prefill=prefill,
            system_prefix=prefix, system_prompt=system_prompt, cot=cot,
        )
        return problem, messages

    jobs = [one(p) for p in problems]
    with ThreadPoolExecutor(max_workers=int(config.get("concurrency", 8))) as pool:
        futures = [(p, pool.submit(backend.complete, m)) for p, m in jobs]
        for done, (problem, future) in enumerate(futures, 1):
            try:
                completion = future.result()
            except BackendError as error:
                raise BackendError(
                    f"{model} / {problem.index}: {error}", status=error.status, body=error.body
                ) from error
            tally.add(completion.text, problem.answer,
                      completion.prompt_tokens, completion.completion_tokens, cot=cot)
            if bar is not None:
                bar.step(done)
            if len(samples) < int(config.get("max_samples", 200)):
                samples.append({
                    "index": problem.index,
                    "gold": problem.answer,
                    "completion": completion.text,
                    "parsed": parse_answer_cot(completion.text) if cot else parse_answer(completion.text),
                })
    return tally, samples


TASK_SIGNAL_PREFIX = {"multiplication": "mult", "arithmetic": "arith", "variable_counting": "varcount"}


def _slug(text, limit):
    """A stage or signal name fragment: letters, digits and underscores only."""
    slug = re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_")
    return slug[:limit]


def phase_name(model, condition, task):
    """One stage per (model, condition, task) cell, so its accuracy is a single series."""
    return f"{_slug(model.split('/')[-1], 20)}__{_slug(condition, 20)}__{_slug(task, 14)}"


def evaluate(config, limit=None, only_models=None, only_conditions=None, report=True):
    counter, counter_name = get_token_counter(config.get("tokenizer", "tiktoken:o200k_base"))
    conditions = build_conditions(config, counter)
    if only_conditions:
        conditions = [c for c in conditions if c["name"] in only_conditions]
    models = [m for m in config["models"] if not only_models or m in only_models]
    if not any(c["name"] == "baseline" for c in conditions):
        raise ValueError("one condition must be named 'baseline'; deltas are measured against it")

    n_problems = int(config.get("n_problems", 1000))
    per_cell = min(limit, n_problems) if limit else n_problems
    shots = int(config.get("shots", 10))
    seed = int(config.get("seed", 12345))
    sampling = config.get("sampling", {})
    cache = ResponseCache(config.get("cache")) if config.get("cache") else None

    tasks = {name: build_task(name, n_problems) for name in config["tasks"]}
    # Few-shot examples come from a separate seeded draw so no eval problem leaks
    examples = {name: build_task(name, shots, seed=seed + 1) for name in config["tasks"]}

    results = {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "paper": config.get("paper", {}),
        "config": {k: v for k, v in config.items() if k != "cache"},
        "tokenizer": counter_name,
        "n_problems": n_problems,
        "limit": limit,
        "shots": shots,
        "seed": seed,
        "conditions": [
            {
                "name": c["name"],
                "filler_type": c["filler"]["type"],
                "filler_words": c["filler"]["words"],
                "filler_tokens_measured": c["filler"]["tokens"],
                "calibration": c["filler"]["calibration"],
                "max_tokens": c["max_tokens"],
                "reasoning": c["reasoning"],
            }
            for c in conditions
        ],
        "cells": [],
    }

    if report:
        orx.plan([
            {"phase": "setup", "of": 1, "unit": "step"},
            *[
                {"phase": phase_name(model, condition["name"], task),
                 "of": per_cell, "unit": "questions"}
                for model in models
                for condition in conditions
                for task in tasks
            ],
        ])
    # The run-level progress object always emits an opening record, so it opens a
    # named one-step stage: an unnamed, totalless stage is what orx reports as a
    # missing instrumentation total.
    run = orx.progress(phase="setup", total=1, unit="step") if report else None
    if run is not None:
        run.step(1)

    for model in models:
        backends = {}

        def backend_for(condition):
            """
            One backend per (max_tokens, reasoning) pair.

            Two CoT conditions at different caps must not share a backend, or the
            second would silently run at the first one's limit. The cap is part of
            the request body, so ResponseCache keys stay distinct per cap.
            """
            key = (condition.get("max_tokens"), condition.get("reasoning"))
            if key not in backends:
                overrides = dict(sampling)
                if key[0] is not None:
                    overrides["max_tokens"] = key[0]
                if key[1] is not None:
                    overrides["reasoning"] = key[1]
                backends[key] = make_backend({**config, "model": model}, overrides, cache)
            return backends[key]

        default_backend = backend_for({"max_tokens": None, "reasoning": None})
        prefill = config.get("prompting", {}).get("prefill", True)
        if getattr(default_backend, "name", None) == "mock":
            results.setdefault("warnings", []).append(
                "mock backend: results exercise the harness and are not scientific"
            )
        model_cells = []
        last_bar = None

        for condition in conditions:
            cot = condition.get("cot", False)
            cell_backend = backend_for(condition)
            for task_name, problems in tasks.items():
                started = time.time()

                def run_condition(bar=None):
                    return run_cell(
                        cell_backend, problems, examples[task_name], condition["filler"],
                        config, model, limit, prefill=prefill, bar=bar, cot=cot,
                    )

                try:
                    if report:
                        with run.phase(phase_name(model, condition["name"], task_name),
                                       total=per_cell, unit="questions") as bar:
                            tally, samples = run_condition(bar)
                        last_bar = bar
                    else:
                        tally, samples = run_condition()
                except BackendError as error:
                    if not prefill or error.status not in (400, 404, 422):
                        raise
                    # the API refuses assistant prefill (Opus 4.6, GPT-5.2 in the paper)
                    results.setdefault("warnings", []).append(
                        f"{model}: assistant prefill rejected (HTTP {error.status}); "
                        "fell back to answer-suffix prompting, which carries selection effects"
                    )
                    prefill = False
                    if report:
                        with run.phase(phase_name(model, condition["name"], task_name),
                                       total=per_cell, unit="questions") as bar:
                            tally, samples = run_condition(bar)
                        last_bar = bar
                    else:
                        tally, samples = run_condition()

                cell = {
                    "model": model,
                    "backend": getattr(cell_backend, "name", config.get("backend", "openrouter")),
                    "task": task_name,
                    "condition": condition["name"],
                    "cot": cot,
                    "filler_type": condition["filler"]["type"],
                    "filler_words": condition["filler"]["words"],
                    "filler_tokens": condition["filler"]["tokens"],
                    "max_tokens": (condition["max_tokens"] if condition["max_tokens"] is not None
                                   else sampling.get("max_tokens")),
                    "reasoning": (condition["reasoning"] if condition["reasoning"] is not None
                                  else sampling.get("reasoning", False)),
                    "prefill": prefill,
                    "seconds": round(time.time() - started, 1),
                    **tally.summary(),
                    "samples": samples,
                }
                baseline = next(
                    (c for c in model_cells if c["task"] == task_name and c["condition"] == "baseline"),
                    None,
                )
                if condition["name"] != "baseline" and baseline is not None:
                    cell["delta_vs_baseline"] = round(cell["accuracy"] - baseline["accuracy"], 2)
                    cell["delta_ci95"] = round(math.hypot(cell["ci95"], baseline["ci95"]), 2)
                model_cells.append(cell)
                results["cells"].append(cell)

                if report:
                    run.row(
                        "cells",
                        task=task_name,
                        condition=condition["name"],
                        model=model.split("/")[-1],
                        max_tokens=cell["max_tokens"],
                        accuracy=cell["accuracy"],
                        ci95=cell["ci95"],
                        answer_rate=cell["answer_rate"],
                        explicit_rate=cell["explicit_rate"],
                        delta=cell.get("delta_vs_baseline"),
                    )
                print(
                    f"[{model}] {task_name:18s} {condition['name']:14s} "
                    f"cap={cell['max_tokens']:<5} "
                    f"acc={cell['accuracy']:6.2f} ±{cell['ci95']:4.2f} "
                    f"(explicit {cell['explicit_rate']:.1f}%, kept {cell['n_kept']}/{cell['n']}, "
                    f"reasoning {cell['reasoning_rate']:.1f}%)",
                    flush=True,
                )

        # Headline signals describe the first model; the cells table carries every model.
        if report and model == models[0] and last_bar is not None:
            summary = {}
            primary = config.get("primary_condition")
            for task_name in tasks:
                cells = {c["condition"]: c for c in model_cells if c["task"] == task_name}
                base = cells.get("baseline")
                filler = cells.get(primary) if primary else None
                prefix = TASK_SIGNAL_PREFIX.get(task_name, _slug(task_name, 8))
                if base is not None:
                    summary[f"{prefix}_baseline"] = base["accuracy"]
                if filler is not None:
                    summary[f"{prefix}_filler"] = filler["accuracy"]
                if base is not None and filler is not None:
                    summary[f"{prefix}_delta"] = filler["delta_vs_baseline"]
            last_bar.step(per_cell, **summary)

    # Every cell is recorded against one named reference condition (the full
    # chain-of-thought cell unless the config says otherwise), so a grid with
    # several CoT caps still has exactly one comparison point per model and task.
    reference = config.get("reference_condition", "cot")
    for cell in results["cells"]:
        ref_cell = next(
            (c for c in results["cells"]
             if c["model"] == cell["model"] and c["task"] == cell["task"]
             and c["condition"] == reference),
            None,
        )
        cell["reference_condition"] = reference
        if ref_cell is not None and cell["condition"] != reference:
            cell["delta_vs_reference"] = round(cell["accuracy"] - ref_cell["accuracy"], 2)
            if reference == "cot":
                cell["delta_vs_cot"] = cell["delta_vs_reference"]

    return results


# -----------------------------------------------------------------------------
# Reporting

def format_summary(results):
    lines = []
    header = f"{'model':32s} {'task':18s} {'condition':16s} {'acc':>7s} {'±':>5s} {'delta':>7s} {'n':>5s}"
    lines.append(header)
    lines.append("-" * len(header))
    for cell in results["cells"]:
        delta = cell.get("delta_vs_baseline")
        lines.append(
            f"{cell['model'][:32]:32s} {cell['task']:18s} {cell['condition']:16s} "
            f"{cell['accuracy']:7.2f} {cell['ci95']:5.2f} "
            f"{(f'{delta:+.2f}' if delta is not None else '—'):>7s} {cell['n_kept']:5d}"
        )
    return "\n".join(lines)


def results_dir():
    base = os.environ.get("ORX_RUN_DIR")
    return os.path.join(base, "results") if base else "results"


def write_results(results):
    directory = results_dir()
    os.makedirs(directory, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(directory, f"filler_eval_{stamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    latest = os.path.join(directory, "filler_eval_latest.json")
    with open(latest, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return path


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="runs/filler_baseline.json")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N problems")
    parser.add_argument("--models", default=None, help="comma-separated subset of configured models")
    parser.add_argument("--conditions", default=None, help="comma-separated subset of condition names")
    parser.add_argument("--dry-run", action="store_true", help="print one prompt per cell and exit")
    parser.add_argument("--no-report", action="store_true",
                        help="skip run progress and measure reporting (for local smoke runs)")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    only_models = args.models.split(",") if args.models else None
    only_conditions = args.conditions.split(",") if args.conditions else None

    if args.dry_run:
        counter, _ = get_token_counter(config.get("tokenizer", "tiktoken:o200k_base"))
        conditions = build_conditions(config, counter)
        shots = int(config.get("shots", 10))
        examples = {name: build_task(name, shots, seed=int(config.get("seed", 12345)) + 1)
                    for name in config["tasks"]}
        for condition in conditions:
            print(f"===== condition {condition['name']} "
                  f"({condition['filler']['tokens']} filler tokens) =====")
            for name in config["tasks"]:
                problem = build_task(name, 1)[0]
                messages = build_messages(
                    problem, examples[name],
                    filler_text=condition["filler"]["text"],
                    prefill=config.get("prompting", {}).get("prefill", True),
                    system_prefix=system_prefix(config, config["models"][0]),
                    cot=condition.get("cot", False),
                )
                print(f"--- {name}: {len(messages)} messages, "
                      f"{sum(len(m['content']) for m in messages)} chars ---")
                print(json.dumps(messages[-2:], indent=2)[:1200])
        return 0

    results = evaluate(config, limit=args.limit, only_models=only_models,
                       only_conditions=only_conditions, report=not args.no_report)
    path = write_results(results)
    print()
    print(format_summary(results))
    print()
    print(f"results written to {path}")
    print("RESULTS_JSON " + json.dumps({"path": path, "cells": len(results["cells"])}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
