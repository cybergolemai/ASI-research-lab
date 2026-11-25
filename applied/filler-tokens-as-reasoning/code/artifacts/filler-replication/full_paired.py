"""Full-N paired statistics reconstructed from the response cache.

The results JSON stores only 200 per-sample records per cell (a disk-size cap),
so the manuscript's paired tests were previously computed over n=200. The cache
itself holds every response, keyed on the exact request body, and the harness is
deterministic: problems come from fixed seeds and the filler calibration is a
pure function of the tokenizer. Rebuilding the request bodies therefore
reproduces the cache keys exactly, and every per-problem outcome can be
recovered.

Regenerate:  uv run --no-project --with tiktoken==0.11.0 python full_paired.py
"""

import json
import math
import os
import sys

import tiktoken

REPO = os.path.expanduser(
    "~/.local/share/openresearch/worktrees/demo_nanochat_v1/"
    "chat_4ba0ff72-3968-4649-bce6-ff790c33b803"
)
CACHE_PATH = os.path.expanduser("~/.cache/nanochat/filler_ste_cot_cache.jsonl")
sys.path.insert(0, REPO)

from nanochat.filler_tasks import build_task  # noqa: E402
from nanochat.filler_tokens import build_filler  # noqa: E402
from nanochat.filler_prompts import build_messages  # noqa: E402
from nanochat.filler_backends import ResponseCache  # noqa: E402
from nanochat.filler_eval import parse_answer, parse_answer_cot  # noqa: E402

MODELS = ("qwen/qwen3-235b-a22b-2507", "qwen/qwen3-30b-a3b-instruct-2507")
TASKS = ("multiplication", "arithmetic")
Z = 1.96

# The exact conditions of the grid run (runs/filler_grid.json). The CoT entry of
# the STE-vs-CoT run predates per-condition caps, but ran at max_tokens=2048
# (commit 58506f1), so its request bodies match the grid's cot condition and the
# same cache entries serve both.
CONDITIONS = (
    ("baseline", None, False, 32, False),
    ("ste_100", 100, False, 32, False),
    ("ste_filler", 200, False, 32, False),
    ("ste_500", 500, False, 32, False),
    ("cot_128", None, True, 128, True),
    ("cot_256", None, True, 256, True),
    ("cot_512", None, True, 512, True),
    ("cot", None, True, 2048, True),
)


def load_cache():
    cache = {}
    with open(CACHE_PATH, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            cache[entry["key"]] = entry
    return cache


def mcnemar_exact(b, c):
    """One-sided exact binomial p for H1: treatment more often correct."""
    n = b + c
    if n == 0:
        return 1.0
    return sum(math.comb(n, k) for k in range(b, n + 1)) / 2 ** n


def main():
    enc = tiktoken.get_encoding("o200k_base")
    counter = lambda text: len(enc.encode(text))  # noqa: E731

    cache = load_cache()
    print(f"cache entries: {len(cache)}")

    fillers = {name: build_filler("ste_dictionary", tokens, counter)["text"]
               for name, tokens in (("ste_100", 100), ("ste_filler", 200),
                                    ("ste_500", 500), ("baseline", 0))}

    outcomes = {}   # (model, task, condition) -> {index: bool}
    misses = 0
    filler_text_by_cond = {}

    for model in MODELS:
        for name, tokens, cot, max_tokens, reasoning in CONDITIONS:
            filler_text = fillers.get(name, "") if not cot else ""
            if name in ("ste_100", "ste_filler", "ste_500", "baseline"):
                filler_text_by_cond[(model, name)] = filler_text
            for task in TASKS:
                problems = build_task(task, 1000)
                correct = {}
                for problem in problems:
                    messages = build_messages(
                        problem, [], filler_text=filler_text, prefill=True,
                        system_prefix="", system_prompt=None, cot=cot,
                    )
                    body = {
                        "model": model, "messages": messages,
                        "temperature": 1.0, "top_p": 0.95,
                        "max_tokens": max_tokens,
                        "usage": {"include": True},
                    }
                    if not reasoning:
                        body["reasoning"] = {"enabled": False}
                    key = ResponseCache.key(body)
                    hit = cache.get(key)
                    if hit is None:
                        misses += 1
                        continue
                    parsed = (parse_answer_cot(hit["text"]) if cot
                              else parse_answer(hit["text"]))
                    correct[problem.index] = parsed is not None and parsed == problem.answer
                outcomes[(model, task, name)] = correct

    total = sum(len(v) for v in outcomes.values())
    print(f"recovered outcomes: {total} / {2 * 2 * 8 * 1000}  (misses: {misses})")

    def paired_stats(model, task, treat, ctrl):
        a, b = outcomes[(model, task, treat)], outcomes[(model, task, ctrl)]
        idx = sorted(set(a) & set(b))
        n = len(idx)
        diffs = [int(a[i]) - int(b[i]) for i in idx]
        mean = 100 * sum(diffs) / n
        sd = math.sqrt(sum((v - sum(diffs) / n) ** 2 for v in diffs) / (n - 1))
        half = 100 * Z * sd / math.sqrt(n)
        plus = sum(1 for i in idx if a[i] and not b[i])
        minus = sum(1 for i in idx if not a[i] and b[i])
        p = mcnemar_exact(plus, minus)
        return mean, half, plus, minus, p, n

    print("\n=== STE vs CoT (235B), full N ===")
    for task in TASKS:
        for treat in ("baseline", "ste_filler"):
            mean, half, plus, minus, p, n = paired_stats(MODELS[0], task, treat, "cot")
            print(f"{task:15s} {treat:11s} vs cot: {mean:+.2f} ± {half:.2f} pp "
                  f"(+{plus}/-{minus}, p={p:.2e}, n={n})")
        mean, half, plus, minus, p, n = paired_stats(MODELS[0], task, "ste_filler", "baseline")
        print(f"{task:15s} ste vs baseline: {mean:+.2f} ± {half:.2f} pp "
              f"(+{plus}/-{minus}, p={p:.2e}, n={n})")

    print("\n=== Model pairs (235B - 30B), full N ===")
    for task in TASKS:
        for name, *_ in CONDITIONS:
            a, b = outcomes[(MODELS[0], task, name)], outcomes[(MODELS[1], task, name)]
            idx = sorted(set(a) & set(b))
            diffs = [int(a[i]) - int(b[i]) for i in idx]
            n = len(diffs)
            mean = 100 * sum(diffs) / n
            sd = math.sqrt(sum((v - sum(diffs) / n) ** 2 for v in diffs) / (n - 1))
            half = 100 * Z * sd / math.sqrt(n)
            print(f"{task:15s} {name:11s}: {mean:+.2f} ± {half:.2f} pp (n={n})")

    paired = {}
    for task in TASKS:
        for treat, ctrl in (("ste_filler", "cot"), ("baseline", "cot"),
                            ("ste_filler", "baseline")):
            mean, half, plus, minus, pval, n = paired_stats(MODELS[0], task, treat, ctrl)
            paired[f"{task}:{treat}-{ctrl}"] = {
                "delta": mean, "ci95": half, "plus": plus, "minus": minus,
                "p": pval, "n": n,
            }

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "full_paired.json"), "w") as handle:
        json.dump({
            "misses": misses,
            "recovered": total,
            "paired": paired,
            "filler_text_200": fillers["ste_filler"],
            "filler_text_100": fillers["ste_100"],
            "filler_text_500": fillers["ste_500"],
        }, handle, indent=2)
    print("\nwrote full_paired.json (incl. exact filler strings)")


if __name__ == "__main__":
    main()
