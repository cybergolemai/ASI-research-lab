"""Paper vs. measured counting-filler accuracy for Qwen3-235B, 10-shot.

Regenerate:  uv run --no-project --with matplotlib --with numpy python comparison.py

Numbers are read from the run's results JSON, not retyped by hand.
"""
import json
import os

import numpy as np
from orx_figstyle import PALETTE, WIDE, family, figure, save, use_style

RESULTS = os.path.expanduser(
    "~/.local/share/openresearch/local-runs/"
    "55bf286c-f6d8-4603-9b1e-48e87164fa66/repo/results/filler_eval_20261122-224806.json"
)

# Paper Table 1, Qwen3-235B row (10-shot, counting 1-100). Mean and 95% CI.
PAPER = {
    "multiplication": {"baseline": (69.6, 0.5), "filler": (69.2, 0.5)},
    "arithmetic": {"baseline": (11.0, 1.0), "filler": (9.7, 0.9)},
}

TASKS = ["multiplication", "arithmetic"]
LABELS = {"multiplication": "4-digit multiplication", "arithmetic": "multi-step arithmetic"}


def measured():
    with open(RESULTS) as handle:
        data = json.load(handle)
    cells = {(c["task"], c["condition"]): c for c in data["cells"]}
    out = {}
    for task in TASKS:
        base = cells[(task, "baseline")]
        fill = cells[(task, "counting_1_100")]
        out[task] = {
            "baseline": (base["accuracy"], base["ci95"]),
            "filler": (fill["accuracy"], fill["ci95"]),
            "delta": fill["delta_vs_baseline"],
            "delta_ci": fill["delta_ci95"],
        }
    return out


def main():
    use_style()
    data = measured()

    # Hue names the source (paper vs this run); shade names the condition.
    series = [
        ("Paper, no filler", "paper", "baseline"),
        ("Paper, 1-100 filler", "paper", "filler"),
        ("This run, no filler", "run", "baseline"),
        ("This run, 1-100 filler", "run", "filler"),
    ]
    colors = {"paper": family("orange", 2), "run": family("blue", 2)}
    shade = {"paper": 0, "run": 0}

    fig, ax = figure(width=WIDE, ratio=0.42)
    x = np.arange(len(TASKS))
    width = 0.19

    for i, (label, group, condition) in enumerate(series):
        offset = (i - (len(series) - 1) / 2) * width
        values, errs = [], []
        for task in TASKS:
            source = PAPER[task] if group == "paper" else data[task]
            mean, ci = source[condition]
            values.append(mean)
            errs.append(ci)
        color = colors[group][shade[group]]
        shade[group] += 1
        bars = ax.bar(
            x + offset, values, width,
            label=label, color=color, edgecolor="black", linewidth=0.4,
            yerr=errs, capsize=1.6, error_kw={"elinewidth": 0.6, "ecolor": "#333333"},
        )
        ax.bar_label(bars, fmt="%.1f", fontsize=5.5, padding=1.5)

    ax.set_xticks(x, [LABELS[t] for t in TASKS])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 88)
    ax.set_yticks(range(0, 81, 20))
    ax.tick_params(axis="x", length=0)
    ax.legend(
        ncol=2, loc="upper right", fontsize=6.0,
        frameon=True, framealpha=1.0, edgecolor="black", borderpad=0.4,
    )

    # The measured delta, placed above each group's tallest bar.
    for task, xpos in zip(TASKS, x):
        tallest = max(PAPER[task]["baseline"][0], data[task]["baseline"][0])
        ax.annotate(
            f"Δ {data[task]['delta']:+.2f} ± {data[task]['delta_ci']:.2f}",
            xy=(xpos, tallest + 7), ha="center", va="center", fontsize=6.5,
            color=PALETTE["blue"],
        )

    save(fig, "filler-comparison")


if __name__ == "__main__":
    main()
