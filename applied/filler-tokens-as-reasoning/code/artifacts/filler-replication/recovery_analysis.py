"""Accuracy recovery across model size, STE filler length and CoT token cap.

Regenerate:  uv run --no-project --with numpy python recovery_analysis.py [RESULTS_JSON]

Writes recovery_cells.csv, recovery_derivatives.csv and recovery_grid.json beside
this script, and prints the markdown tables used in the report.

Derivatives are finite differences of cell accuracies, with 95% CIs from the
binomial standard errors (cell ci95 / 1.96) combined in quadrature: the cells are
scored on the same 1,000 problems, so that interval is conservative. Where a
paired comparison is possible (two models, same condition), the 200 stored
per-sample outcomes give the tighter paired interval as a cross-check.
"""

import csv
import json
import math
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS = os.path.expanduser(
    "~/.local/share/openresearch/local-runs/"
    "09ce0925-8c04-4766-9dce-bf89bd220203/repo/results/filler_eval_latest.json"
)
REPO = os.path.expanduser(
    "~/.local/share/openresearch/worktrees/demo_nanochat_v1/"
    "chat_4ba0ff72-3968-4649-bce6-ff790c33b803"
)

TASKS = ("multiplication", "arithmetic")
TASK_LABEL = {"multiplication": "4-digit multiplication", "arithmetic": "multi-step arithmetic"}
STE_CONDITIONS = (("baseline", 0), ("ste_100", 100), ("ste_filler", 200), ("ste_500", 500))
COT_CAPS = (("cot_128", 128), ("cot_256", 256), ("cot_512", 512), ("cot", 2048))
Z = 1.96
GIB = 1024 ** 3


def cell_se(cell):
    return cell["ci95"] / Z


def ci_text(value, half, digits=2):
    return f"{value:+.{digits}f} ± {half:.{digits}f}"


def slope(a, b, dx):
    """Difference quotient and its quadrature SE."""
    value = (b["accuracy"] - a["accuracy"]) / dx
    se = math.hypot(cell_se(a), cell_se(b)) / abs(dx)
    return value, se


def wls(x, y, se, degree):
    """Weighted least squares; returns coefficients (highest power first) and R^2.

    A cell at exactly 0% accuracy has a zero binomial SE, which would give an
    infinite weight. Floor the SE at 0.05pp so the fit stays finite; a zero-SE
    point is still the most trusted point in the fit.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = 1.0 / np.maximum(np.asarray(se, dtype=float), 0.05) ** 2
    design = np.vstack([x ** (degree - i) for i in range(degree + 1)]).T
    coef, *_ = np.linalg.lstsq(design * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)
    pred = design @ coef
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    return coef, r2


def bootstrap_ratio(numerator, denominator, draws=4000, seed=11):
    """95% CI for a ratio of two noisy slopes (parametric normal bootstrap)."""
    rng = random.Random(seed)
    a, a_se = numerator
    b, b_se = denominator
    out = []
    for _ in range(draws):
        av = rng.gauss(a, a_se)
        bv = rng.gauss(b, b_se)
        if abs(bv) > 1e-9:
            out.append(av / bv)
    out.sort()
    if not out:
        return float("nan"), float("nan")
    lo = out[int(0.025 * len(out))]
    hi = out[int(0.975 * len(out)) - 1]
    return lo, hi


def load(path):
    with open(path) as handle:
        return json.load(handle)


def problem_token_means(n_problems):
    sys.path.insert(0, REPO)
    import tiktoken
    from nanochat.filler_tasks import build_task

    enc = tiktoken.get_encoding("o200k_base")
    means = {}
    for task in TASKS:
        problems = build_task(task, n_problems)
        lengths = [len(enc.encode(p.prompt)) for p in problems]
        means[task] = sum(lengths) / len(lengths)
    return means


def kv_geometry(config):
    out = {}
    for model, spec in config["model_geometry"].items():
        out[model] = 2 * spec["layers"] * spec["kv_heads"] * spec["head_dim"] * 2
    return out


def build_cells(results, problem_means, kv_bytes):
    cells = {}
    for cell in results["cells"]:
        n = cell["n"]
        t_prompt = cell["prompt_tokens"] / n
        t_comp = cell["completion_tokens"] / n
        shared = t_prompt - problem_means[cell["task"]]
        t_kv = t_prompt + t_comp
        reads = t_comp * t_prompt + t_comp * (t_comp - 1) / 2
        row = {
            "model": cell["model"],
            "task": cell["task"],
            "condition": cell["condition"],
            "max_tokens": cell["max_tokens"],
            "filler_tokens": cell["filler_tokens"],
            "n": n,
            "accuracy": cell["accuracy"],
            "ci95": cell["ci95"],
            "se": cell_se(cell),
            "answer_rate": cell["answer_rate"],
            "explicit_rate": cell["explicit_rate"],
            "prompt_tokens": cell["prompt_tokens"],
            "completion_tokens": cell["completion_tokens"],
            "t_prompt": t_prompt,
            "t_comp": t_comp,
            "t_kv": t_kv,
            "shared_prefix": shared,
            "unshared": problem_means[cell["task"]] + t_comp,
            "kv_reads": reads,
            "seconds": cell["seconds"],
            "seconds_per_query": cell["seconds"] / n,
            "kv_bytes_per_token": kv_bytes[cell["model"]],
            "peak_kv_mib": t_kv * kv_bytes[cell["model"]] / 1024 ** 2,
            "occupancy_mib_s": t_kv * kv_bytes[cell["model"]] / 1024 ** 2 * cell["seconds"] / n,
            "samples": cell["samples"],
        }
        cells[(cell["model"], cell["task"], cell["condition"])] = row
    return cells


def paired_model_delta(cells, model_a, model_b, task, condition):
    """Paired accuracy difference over the stored samples (same problem indices)."""
    a = cells[(model_a, task, condition)]["samples"]
    b = cells[(model_b, task, condition)]["samples"]
    left = {s["index"]: s["parsed"] == s["gold"] for s in a}
    right = {s["index"]: s["parsed"] == s["gold"] for s in b}
    idx = sorted(set(left) & set(right))
    diffs = [int(left[i]) - int(right[i]) for i in idx]
    n = len(diffs)
    mean = sum(diffs) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in diffs) / (n - 1))
    se = sd / math.sqrt(n)
    return 100 * mean, 100 * Z * se, n


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RESULTS
    results = load(path)
    config = results["config"]
    kv_bytes = kv_geometry(config)
    problem_means = problem_token_means(results["n_problems"])
    cells = build_cells(results, problem_means, kv_bytes)

    models = []
    for row in cells.values():
        if row["model"] not in models:
            models.append(row["model"])
    big, small = models[0], models[1]
    label = {big: "235B-A22B", small: "30B-A3B"}
    totals = {m: s["total_params_b"] for m, s in config["model_geometry"].items()}
    actives = {m: s["active_params_b"] for m, s in config["model_geometry"].items()}

    print(f"# results: {path}")
    print(f"# cells: {len(cells)}  models: {[label[m] for m in models]}")

    # ---------------- cell table ----------------
    rows = []
    for (model, task, cond), row in cells.items():
        rows.append({
            "model": label[model], "task": task, "condition": cond,
            "filler_tokens": row["filler_tokens"], "max_tokens": row["max_tokens"],
            "accuracy": row["accuracy"], "ci95": row["ci95"],
            "explicit_rate": row["explicit_rate"], "answer_rate": row["answer_rate"],
            "t_prompt": round(row["t_prompt"], 2), "t_comp": round(row["t_comp"], 2),
            "peak_kv_mib": round(row["peak_kv_mib"], 2),
            "kv_reads": round(row["kv_reads"], 1),
            "seconds_per_query": round(row["seconds_per_query"], 3),
        })
    rows.sort(key=lambda r: (r["model"], r["task"], r["condition"]))
    with open(os.path.join(HERE, "recovery_cells.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # ---------------- A: parameter sensitivity ----------------
    dlog_total = math.log10(totals[big] / totals[small])
    dlog_active = math.log10(actives[big] / actives[small])
    dparams_total = (totals[big] - totals[small]) / 10.0
    dparams_active = (actives[big] - actives[small]) / 10.0

    deriv = []
    for task in TASKS:
        for cond, length in STE_CONDITIONS:
            a, b = cells[(big, task, cond)], cells[(small, task, cond)]
            value, se = slope(b, a, dlog_total)
            _, se_act = slope(b, a, dlog_active)
            pair, pair_half, pair_n = paired_model_delta(cells, big, small, task, cond)
            deriv.append({
                "axis": "params_total_log", "task": task, "condition": cond,
                "level": length, "value": value, "se": se, "unit": "pp/decade",
                "paired": pair, "paired_ci95": pair_half, "paired_n": pair_n,
            })
            deriv.append({
                "axis": "params_active_log", "task": task, "condition": cond,
                "level": length, "value": (a["accuracy"] - b["accuracy"]) / dlog_active,
                "se": math.hypot(cell_se(a), cell_se(b)) / dlog_active,
                "unit": "pp/decade", "paired": pair, "paired_ci95": pair_half, "paired_n": pair_n,
            })
            deriv.append({
                "axis": "params_total_10b", "task": task, "condition": cond,
                "level": length, "value": (a["accuracy"] - b["accuracy"]) / dparams_total,
                "se": math.hypot(cell_se(a), cell_se(b)) / dparams_total,
                "unit": "pp/10B total", "paired": pair, "paired_ci95": pair_half, "paired_n": pair_n,
            })
            deriv.append({
                "axis": "params_active_10b", "task": task, "condition": cond,
                "level": length, "value": (a["accuracy"] - b["accuracy"]) / dparams_active,
                "se": math.hypot(cell_se(a), cell_se(b)) / dparams_active,
                "unit": "pp/10B active", "paired": pair, "paired_ci95": pair_half, "paired_n": pair_n,
            })
        for cond, cap in COT_CAPS:
            a, b = cells[(big, task, cond)], cells[(small, task, cond)]
            pair, pair_half, pair_n = paired_model_delta(cells, big, small, task, cond)
            deriv.append({
                "axis": "params_total_log", "task": task, "condition": cond,
                "level": cap, "value": (a["accuracy"] - b["accuracy"]) / dlog_total,
                "se": math.hypot(cell_se(a), cell_se(b)) / dlog_total,
                "unit": "pp/decade", "paired": pair, "paired_ci95": pair_half, "paired_n": pair_n,
            })

    print("\n## A. Parameter sensitivity (pp per decade of parameters)\n")
    print("| task | condition | 235B − 30B | ∂Acc/∂log₁₀N_total | ∂Acc/∂log₁₀N_active | paired 95% CI |")
    print("| --- | --- | --- | --- | --- | --- |")
    for task in TASKS:
        for cond, _ in STE_CONDITIONS + COT_CAPS:
            a, b = cells[(big, task, cond)], cells[(small, task, cond)]
            value = (a["accuracy"] - b["accuracy"]) / dlog_total
            se = math.hypot(cell_se(a), cell_se(b)) / dlog_total
            se_a = se * dlog_total / dlog_active
            pair, pair_half, pair_n = paired_model_delta(cells, big, small, task, cond)
            print(f"| {task} | {cond} | {a['accuracy'] - b['accuracy']:+.1f} | "
                  f"{value:+.1f} ± {Z * se:.1f} | {value * dlog_total / dlog_active:+.1f} ± {Z * se_a:.1f} | "
                  f"{pair:+.1f} ± {pair_half:.1f} (n={pair_n}) |")

    # ---------------- B: STE length ----------------
    print("\n## B. STE filler-length sensitivity\n")
    fits = {}
    for model in models:
        for task in TASKS:
            xs = [length for _, length in STE_CONDITIONS]
            ys = [cells[(model, task, c)]["accuracy"] for c, _ in STE_CONDITIONS]
            ses = [cell_se(cells[(model, task, c)]) for c, _ in STE_CONDITIONS]
            coef_quad, r2_quad = wls(xs, ys, ses, 2)
            coef_lin, r2_lin = wls(xs, ys, ses, 1)
            pos = [(l, y, s) for l, y, s in zip(xs, ys, ses) if l > 0]
            coef_log, r2_log = wls([math.log10(p[0]) for p in pos], [p[1] for p in pos],
                                   [p[2] for p in pos], 1)
            fits[(model, task)] = (coef_quad, r2_quad, coef_lin, r2_lin, coef_log, r2_log)

    print("| model | task | [0,100] | [100,200] | [200,500] | linear fit | quadratic fit | log fit (L>0) |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for model in models:
        for task in TASKS:
            segs = []
            for (c0, l0), (c1, l1) in zip(STE_CONDITIONS, STE_CONDITIONS[1:]):
                v, s = slope(cells[(model, task, c0)], cells[(model, task, c1)], l1 - l0)
                segs.append(f"{v * 100:+.1f} ± {Z * s * 100:.1f}")
                deriv.append({
                    "axis": "ste_length", "task": task, "condition": f"{c0}->{c1}",
                    "model": model, "level": (l0 + l1) / 2,
                    "value": v * 100, "se": s * 100, "unit": "pp/100 tokens",
                })
            coef_quad, r2_quad, coef_lin, r2_lin, coef_log, r2_log = fits[(model, task)]
            print(f"| {label[model]} | {task} | {segs[0]} | {segs[1]} | {segs[2]} | "
                  f"{coef_lin[0] * 100:+.2f} pp/100 (R²={r2_lin:.2f}) | "
                  f"curv {coef_quad[0]:+.4f} (R²={r2_quad:.2f}) | "
                  f"{coef_log[0]:+.1f} pp/decade (R²={r2_log:.2f}) |")
            deriv.append({
                "axis": "ste_length_linear", "task": task, "condition": "0-500",
                "model": model, "level": 250,
                "value": coef_lin[0] * 100, "se": float("nan"), "unit": "pp/100 tokens",
                "r2": r2_lin,
            })

    # ---------------- C: CoT cap ----------------
    print("\n## C. CoT token-cap sensitivity\n")
    print("| model | task | accuracy at each cap | realized completion tokens | wrote the Answer cue |")
    print("| --- | --- | --- | --- | --- |")
    for model in models:
        for task in TASKS:
            caps = [cap for _, cap in COT_CAPS]
            ys = [cells[(model, task, c)]["accuracy"] for c, _ in COT_CAPS]
            realized = [cells[(model, task, c)]["t_comp"] for c, _ in COT_CAPS]
            explicit = [cells[(model, task, c)]["explicit_rate"] for c, _ in COT_CAPS]
            print(f"| {label[model]} | {task} | "
                  + " | ".join(f"{c}:{y:.0f}%" for c, y in zip(caps, ys))
                  + " | " + " | ".join(f"{r:.0f}" for r in realized)
                  + " | " + " | ".join(f"{e:.0f}%" for e in explicit) + " |")
    print("\n| model | task | cap interval | pp/100 cap tokens | pp/100 realized tokens |")
    print("| --- | --- | --- | --- | --- |")
    for model in models:
        for task in TASKS:
            for (c0, l0), (c1, l1) in zip(COT_CAPS, COT_CAPS[1:]):
                a, b = cells[(model, task, c0)], cells[(model, task, c1)]
                v_cap, s_cap = slope(a, b, (l1 - l0) / 100)
                dr = (b["t_comp"] - a["t_comp"]) / 100
                if abs(dr) > 1e-6:
                    v_real = (b["accuracy"] - a["accuracy"]) / dr
                    s_real = math.hypot(cell_se(a), cell_se(b)) / abs(dr)
                    real = f"{v_real:+.1f} ± {Z * s_real:.1f}"
                else:
                    real = "—"
                print(f"| {label[model]} | {task} | {c0}→{c1} | {v_cap:+.1f} ± {Z * s_cap:.1f} | {real} |")
                deriv.append({
                    "axis": "cot_cap", "task": task, "condition": f"{c0}->{c1}",
                    "model": model, "level": (l0 + l1) / 2,
                    "value": v_cap, "se": s_cap, "unit": "pp/100 cap tokens",
                })
            # global fit on realized tokens across the four caps
            xs = [cells[(model, task, c)]["t_comp"] for c, _ in COT_CAPS]
            ys = [cells[(model, task, c)]["accuracy"] for c, _ in COT_CAPS]
            ses = [cell_se(cells[(model, task, c)]) for c, _ in COT_CAPS]
            coef, r2 = wls(xs, ys, ses, 1)
            print(f"| {label[model]} | {task} | 128→2048 fit | "
                  f"({coef[0] * 100:+.2f}/100 realized, R²={r2:.2f}) | |")
            deriv.append({
                "axis": "cot_realized_linear", "task": task, "condition": "128-2048",
                "model": model, "level": sum(xs) / len(xs),
                "value": coef[0] * 100, "se": float("nan"), "unit": "pp/100 realized tokens",
                "r2": r2,
            })

    # ---------------- cross-partial: does size amplify STE recovery? ----------------
    print("\n## Cross-partial: ∂²Acc / ∂log₁₀N ∂L_STE\n")
    print("| task | interval | 235B slope | 30B slope | difference (pp/100 tokens per decade) |")
    print("| --- | --- | --- | --- | --- |")
    for task in TASKS:
        for (c0, l0), (c1, l1) in zip(STE_CONDITIONS, STE_CONDITIONS[1:]):
            v_big, s_big = slope(cells[(big, task, c0)], cells[(big, task, c1)], l1 - l0)
            v_small, s_small = slope(cells[(small, task, c0)], cells[(small, task, c1)], l1 - l0)
            diff = (v_small - v_big) * 100 / dlog_total
            se = math.hypot(s_small, s_big) * 100 / dlog_total
            print(f"| {task} | {c0}→{c1} | {v_big * 100:+.1f} | {v_small * 100:+.1f} | "
                  f"{diff:+.1f} ± {Z * se:.1f} |")
            deriv.append({
                "axis": "cross_partial", "task": task, "condition": f"{c0}->{c1}",
                "level": (l0 + l1) / 2, "value": diff, "se": se,
                "unit": "pp/100 tokens per decade",
            })

    # ---------------- MRTS ----------------
    print("\n## Marginal rates of technical substitution\n")
    for task in TASKS:
        ste_big, ste_big_se = slope(cells[(big, task, "baseline")], cells[(big, task, "ste_500")], 500)
        ste_small, ste_small_se = slope(cells[(small, task, "baseline")], cells[(small, task, "ste_500")], 500)
        cot_hi, cot_hi_se = slope(cells[(big, task, "cot_512")], cells[(big, task, "cot")], 1536)
        pn_big = (cells[(big, task, "baseline")]["accuracy"] - cells[(small, task, "baseline")]["accuracy"]) / dlog_total
        pn_se = math.hypot(cell_se(cells[(big, task, "baseline")]),
                           cell_se(cells[(small, task, "baseline")])) / dlog_total
        lo, hi = bootstrap_ratio((ste_big * 100, ste_big_se * 100), (cot_hi * 100, cot_hi_se * 100))
        lo_n, hi_n = bootstrap_ratio((ste_big * 100, ste_big_se * 100), (pn_big, pn_se))
        print(f"- **{task}**: ∂Acc/∂L_STE = {ste_big * 100:+.2f} pp/100 tok (235B), "
              f"{ste_small * 100:+.2f} (30B); ∂Acc/∂L_COT = {cot_hi * 100:+.3f} pp/100 tok "
              f"over 512→2048; ∂Acc/∂log₁₀N = {pn_big:+.1f} pp/decade at baseline.")
        print(f"  - MRTS(L_STE, L_COT) = {ste_big / cot_hi:.2f} CoT tokens per STE token "
              f"[95% CI {lo:.2f}, {hi:.2f}]")
        print(f"  - MRTS(L_STE, log₁₀N) = {ste_big * 100 / pn_big:.1f} STE tokens per decade "
              f"[95% CI {lo_n:.1f}, {hi_n:.1f}]")

    # ---------------- Pareto ----------------
    print("\n## Pareto frontier: accuracy against peak KV per query\n")
    for task in TASKS:
        pts = [(label[m], c, cells[(m, task, c)]) for m in models for c, _ in STE_CONDITIONS + COT_CAPS]
        dominated = []
        for name, cond, row in pts:
            better = [o for o in pts
                      if o[2]["accuracy"] >= row["accuracy"]
                      and o[2]["peak_kv_mib"] <= row["peak_kv_mib"]
                      and (o[2]["accuracy"] > row["accuracy"] or o[2]["peak_kv_mib"] < row["peak_kv_mib"])]
            if better:
                dominated.append((name, cond))
        front = [(n, c) for n, c, _ in pts if (n, c) not in dominated]
        print(f"- **{task}** frontier: " + ", ".join(f"{n}/{c}" for n, c in front))
        print(f"  dominated: " + ", ".join(f"{n}/{c}" for n, c in dominated))

    # ---------------- write derivatives ----------------
    with open(os.path.join(HERE, "recovery_derivatives.csv"), "w", newline="") as handle:
        fieldnames = sorted({k for row in deriv for k in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in deriv:
            writer.writerow({k: row.get(k) for k in fieldnames})

    summary = {
        "results": path,
        "models": {label[m]: {"kv_bytes_per_token": kv_bytes[m], **config["model_geometry"][m]}
                   for m in models},
        "dlog10_total": dlog_total,
        "dlog10_active": dlog_active,
        "cells": rows,
        "derivatives": deriv,
    }
    with open(os.path.join(HERE, "recovery_grid.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"\nwrote {os.path.join(HERE, 'recovery_cells.csv')}, "
          f"recovery_derivatives.csv, recovery_grid.json")


if __name__ == "__main__":
    main()
