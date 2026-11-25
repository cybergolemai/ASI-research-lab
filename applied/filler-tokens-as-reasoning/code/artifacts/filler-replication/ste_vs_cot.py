"""0-shot baseline vs STE filler vs full CoT on Qwen3-235B.

Regenerate:  uv run --no-project --with matplotlib --with numpy python ste_vs_cot.py

Reads the run's results JSON; the paired deltas come from the stored samples.
"""
import json
import math
import os

import numpy as np
from orx_figstyle import BASELINE, PALETTE, WIDE, family, figure, save, use_style

RESULTS = os.path.expanduser(
    "~/.local/share/openresearch/local-runs/"
    "dbcdb225-18b3-4454-8e05-95bc8c79fc13/repo/results/filler_eval_20261123-073740.json"
)

TASKS = ["multiplication", "arithmetic"]
LABELS = {"multiplication": "4-digit multiplication", "arithmetic": "multi-step arithmetic"}


def load():
    with open(RESULTS) as handle:
        data = json.load(handle)
    cells = {(c["task"], c["condition"]): c for c in data["cells"]}
    out = {}
    for task in TASKS:
        row = {}
        for cond in ("baseline", "ste_filler", "cot"):
            cell = cells[(task, cond)]
            row[cond] = (cell["accuracy"], cell["ci95"])
        # paired STE vs CoT delta from the stored samples
        base = {s["index"]: s["parsed"] == s["gold"] for s in cells[(task, "ste_filler")]["samples"]}
        cot = {s["index"]: s["parsed"] == s["gold"] for s in cells[(task, "cot")]["samples"]}
        idx = sorted(set(base) & set(cot))
        diffs = [int(base[i]) - int(cot[i]) for i in idx]
        mean = sum(diffs) / len(diffs)
        sd = math.sqrt(sum((v - mean) ** 2 for v in diffs) / (len(diffs) - 1)) if len(diffs) > 1 else 0.0
        se = sd / math.sqrt(len(diffs))
        row["ste_minus_cot"] = (100 * mean, 100 * 1.96 * se)
        out[task] = row
    return out


def main():
    use_style()
    data = load()

    conditions = [
        ("Baseline (no CoT)", "baseline", BASELINE),
        ("STE filler", "ste_filler", family("orange", 1)[0]),
        ("Full CoT", "cot", family("blue", 1)[0]),
    ]

    fig, ax = figure(width=WIDE, ratio=0.44)
    x = np.arange(len(TASKS))
    width = 0.24

    for i, (label, key, color) in enumerate(conditions):
        offset = (i - (len(conditions) - 1) / 2) * width
        values = [data[t][key][0] for t in TASKS]
        errs = [data[t][key][1] for t in TASKS]
        bars = ax.bar(
            x + offset, values, width, label=label, color=color,
            edgecolor="black", linewidth=0.4,
            yerr=errs, capsize=1.6, error_kw={"elinewidth": 0.6, "ecolor": "#333333"},
        )
        ax.bar_label(bars, fmt="%.1f", fontsize=5.5, padding=1.5)

    ax.set_xticks(x, [LABELS[t] for t in TASKS])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 108)
    ax.set_yticks(range(0, 101, 20))
    ax.tick_params(axis="x", length=0)
    ax.legend(ncol=3, loc="upper center", fontsize=6.0, frameon=False,
              bbox_to_anchor=(0.5, 1.02))

    for task, xpos in zip(TASKS, x):
        delta, ci = data[task]["ste_minus_cot"]
        ax.annotate(
            f"STE − CoT = {delta:+.1f} ± {ci:.1f}",
            xy=(xpos, 102), ha="center", va="center", fontsize=6.5,
            color=PALETTE["red"],
        )

    save(fig, "ste-vs-cot")


if __name__ == "__main__":
    main()
