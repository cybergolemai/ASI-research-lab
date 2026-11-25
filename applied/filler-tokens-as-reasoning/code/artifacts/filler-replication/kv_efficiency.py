"""KV-cache and decode-efficiency accounting for the STE-vs-CoT run.

Regenerate:  uv run --with tiktoken python kv_efficiency.py

Reads the run's results JSON, regenerates the same 1,000 problems to measure
prompt tokens, and re-tokenizes the 200 stored completions per cell to get a
sampling distribution for completion length.

Assumption block (Qwen3-235B-A22B-Instruct-2507 config.json, fetched 2026-11-23):
  num_hidden_layers = 94, num_key_value_heads = 4, head_dim = 128,
  torch_dtype = bfloat16  ->  2 * 94 * 4 * 128 * 2 B = 192,512 B/token (~188 KiB)
"""

import json
import math
import os
import random
import statistics
import sys

import tiktoken

RESULTS = os.path.expanduser(
    "~/.local/share/openresearch/local-runs/"
    "dbcdb225-18b3-4454-8e05-95bc8c79fc13/repo/results/filler_eval_20261123-073740.json"
)

REPO = os.path.expanduser(
    "~/.local/share/openresearch/worktrees/demo_nanochat_v1/"
    "chat_4ba0ff72-3968-4649-bce6-ff790c33b803"
)

LAYERS, KV_HEADS, HEAD_DIM, BYTES = 94, 4, 128, 2
BYTES_PER_TOKEN = 2 * LAYERS * KV_HEADS * HEAD_DIM * BYTES  # K and V
GIB = 1024 ** 3

CONDITIONS = ("baseline", "ste_filler", "cot")
TASKS = ("multiplication", "arithmetic")


def mean_ci(values):
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0, 0.0
    sd = statistics.stdev(values)
    half = 1.96 * sd / math.sqrt(n)
    return mean, sd, half


def kv_reads(t_prompt, t_completion):
    """Sum_{t=0}^{T_c-1} (T_p + t): KV tokens read to attend over the context."""
    return t_completion * t_prompt + t_completion * (t_completion - 1) / 2


def bootstrap_ratio(a, b, draws=4000, seed=7):
    """95% CI for mean(a)/mean(b) from paired per-query completion lengths."""
    rng = random.Random(seed)
    n = len(a)
    ratios = []
    for _ in range(draws):
        idx = [rng.randrange(n) for _ in range(n)]
        num = sum(a[i] for i in idx)
        den = sum(b[i] for i in idx)
        if den:
            ratios.append(num / den)
    ratios.sort()
    return ratios[int(0.025 * draws)], ratios[int(0.975 * draws)]


def mcnemar_one_sided(treatment, control):
    """Exact binomial p for H1: treatment more often correct than control."""
    b = sum(1 for t, c in zip(treatment, control) if t and not c)
    c_ = sum(1 for t, c in zip(treatment, control) if c and not t)
    n = b + c_
    if n == 0:
        return b, c_, 1.0
    # P(X >= b) with X ~ Binomial(n, 1/2)
    tail = sum(math.comb(n, k) for k in range(b, n + 1)) / 2 ** n
    return b, c_, tail


def main():
    sys.path.insert(0, REPO)
    from nanochat.filler_tasks import build_task

    enc = tiktoken.get_encoding("o200k_base")
    with open(RESULTS) as handle:
        data = json.load(handle)

    cells = {(c["task"], c["condition"]): c for c in data["cells"]}
    problems = {t: build_task(t, data["n_problems"]) for t in TASKS}
    problem_tokens = {t: [len(enc.encode(p.prompt)) for p in problems[t]] for t in TASKS}

    rows = {}
    for task in TASKS:
        n = cells[(task, "baseline")]["n"]
        ptoks = problem_tokens[task]
        mean_problem = sum(ptoks) / len(ptoks)
        for cond in CONDITIONS:
            cell = cells[(task, cond)]
            t_prompt = cell["prompt_tokens"] / n
            t_comp = cell["completion_tokens"] / n
            seconds = cell["seconds"]
            # sampling distribution of completion length from the stored samples
            lengths = [len(enc.encode(s["completion"])) for s in cell["samples"]]
            lmean, _, lhalf = mean_ci(lengths)
            rel_half = lhalf / lmean if lmean else 0.0
            rows[(task, cond)] = {
                "n": n,
                "acc": cell["accuracy"],
                "acc_ci": cell["ci95"],
                "t_prompt": t_prompt,
                "problem": mean_problem,
                "shared": t_prompt - mean_problem,
                "t_comp": t_comp,
                "t_comp_half": t_comp * rel_half,
                "t_kv": t_prompt + t_comp,
                "unshared": mean_problem + t_comp,
                "reads": kv_reads(t_prompt, t_comp),
                "seconds": seconds,
                "slot": seconds / n,
                "lengths": lengths,
            }

    for task in TASKS:
        print(f"\n=== {task} (N={rows[(task, 'baseline')]['n']}) ===")
        print(f"{'condition':<12}{'T_prompt':>10}{'T_comp':>9}{'T_KV':>9}"
              f"{'shared':>9}{'unshared':>10}{'KV reads':>12}{'s/query':>10}"
              f"{'GB tok-s':>10}{'acc':>8}")
        for cond in CONDITIONS:
            r = rows[(task, cond)]
            print(f"{cond:<12}{r['t_prompt']:>10.1f}{r['t_comp']:>9.1f}{r['t_kv']:>9.1f}"
                  f"{r['shared']:>9.1f}{r['unshared']:>10.1f}{r['reads']:>12,.0f}"
                  f"{r['slot']:>10.2f}{r['t_kv'] * r['slot'] * BYTES_PER_TOKEN / 1e9:>10.4f}"
                  f"{r['acc']:>8.2f}")

        base, ste, cot = (rows[(task, c)] for c in CONDITIONS)
        for name, a, b in (("ste/cot", ste, cot), ("ste/baseline", ste, base)):
            lo, hi = bootstrap_ratio(a["lengths"], b["lengths"])
            text = (sum(a["lengths"]) / len(a["lengths"])) / (sum(b["lengths"]) / len(b["lengths"]))
            api = a["t_comp"] / b["t_comp"]
            red = (f"(1/{b['reads'] / a['reads']:.0f})" if a["reads"] < b["reads"]
                   else f"({a['reads'] / b['reads']:.1f}x fewer)")
            print(f"  {name:<13} T_KV x{a['t_kv'] / b['t_kv']:.3f}   "
                  f"reads x{a['reads'] / b['reads']:.4f} {red}   "
                  f"time x{a['slot'] / b['slot']:.3f}   "
                  f"comp ratio {api:.4f} CI[{api * lo / text:.4f}, {api * hi / text:.4f}]")
        print(f"  acc per 1e3 KV reads: "
              f"baseline {base['acc'] / (base['reads'] / 1e3):.1f}  "
              f"ste {ste['acc'] / (ste['reads'] / 1e3):.1f}  "
              f"cot {cot['acc'] / (cot['reads'] / 1e3):.2f}")
        red = cot["reads"] / ste["reads"]
        print(f"  STE accuracy cost: {ste['acc'] - cot['acc']:+.2f} pp for a "
              f"{red:.0f}x KV-read reduction "
              f"({(ste['acc'] - cot['acc']) / (red - 1):.3f} pp per 1x)")

    # ---------------- hypothesis tests ----------------
    for task in TASKS:
        base = {s["index"]: s for s in cells[(task, "baseline")]["samples"]}
        ste = {s["index"]: s for s in cells[(task, "ste_filler")]["samples"]}
        idx = sorted(set(base) & set(ste))
        ok_b = [base[i]["parsed"] == base[i]["gold"] for i in idx]
        ok_s = [ste[i]["parsed"] == ste[i]["gold"] for i in idx]
        b, c_, p = mcnemar_one_sided(ok_s, ok_b)
        diffs = [int(s) - int(x) for s, x in zip(ok_s, ok_b)]
        m, _, half = mean_ci(diffs)
        print(f"\n=== uplift test, {task} (paired n={len(idx)}) ===")
        print(f"  STE-only correct {b}, baseline-only correct {c_}, "
              f"exact one-sided p={p:.4f}")
        print(f"  paired delta {100 * m:+.2f} pp, 95% CI +/-{100 * half:.2f}")

    # ---------------- KV scaling ----------------
    print("\n=== KV footprint: 16-way concurrency (bf16 KV, 188 KiB/token) ===")
    conc = 16
    for task in TASKS:
        for cond in CONDITIONS:
            r = rows[(task, cond)]
            naive = r["t_kv"] * conc * BYTES_PER_TOKEN / 1e6
            paged = (r["shared"] + conc * r["unshared"]) * BYTES_PER_TOKEN / 1e6
            print(f"  {task:<15}{cond:<12}"
                  f"resident @16 {naive:7.1f} MB   "
                  f"with prefix cache {paged:7.1f} MB   "
                  f"per-token {BYTES_PER_TOKEN / 1024:.0f} KiB")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kv_efficiency.json")
    serializable = {
        f"{t}/{c}": {k: v for k, v in r.items() if k != "lengths"}
        for (t, c), r in rows.items()
    }
    serializable["geometry"] = {
        "layers": LAYERS, "kv_heads": KV_HEADS, "head_dim": HEAD_DIM,
        "dtype_bytes": BYTES, "kv_bytes_per_token": BYTES_PER_TOKEN,
    }
    with open(out, "w") as handle:
        json.dump(serializable, handle, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
