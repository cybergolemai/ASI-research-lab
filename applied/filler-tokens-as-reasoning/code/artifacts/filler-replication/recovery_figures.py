"""Publication figures for the accuracy-recovery grid.

Regenerate:  uv run --no-project --with matplotlib --with numpy python recovery_figures.py

Reads recovery_grid.json (written by recovery_analysis.py) and produces:
  recovery-pareto.{svg,pdf}      accuracy against peak KV MiB and decode KV reads
  recovery-sensitivity.{svg,pdf} response curves across N, L_STE and L_COT
"""

import json
import math
import os

import numpy as np
from orx_figstyle import BASELINE, PALETTE, WIDE, figure_grid, pareto_front, save, use_style

HERE = os.path.dirname(os.path.abspath(__file__))

MODELS = [("235B-A22B", "235B / 22B active"), ("30B-A3B", "30B / 3.3B active")]
TASKS = [("multiplication", "4-digit multiplication", "o"), ("arithmetic", "multi-step arithmetic", "s")]
STE = [("baseline", 0), ("ste_100", 100), ("ste_filler", 200), ("ste_500", 500)]
COT = [("cot_128", 128), ("cot_256", 256), ("cot_512", 512), ("cot", 2048)]


def load():
    with open(os.path.join(HERE, "recovery_grid.json")) as handle:
        data = json.load(handle)
    cells = {(c["model"], c["task"], c["condition"]): c for c in data["cells"]}
    return data, cells


def pareto_figure(data, cells):
    """Accuracy against KV cost, one panel per task and per cost metric.

    Two families per panel: the STE sweep (baseline -> ste_500) and the CoT cap
    sweep (128 -> 2048), drawn as separate curves so the caps are never joined to
    the filler points. Solid = CoT, dashed = STE.
    """
    fig, axes = figure_grid(nrows=2, ncols=2, width=WIDE, ratio=0.62, sharey=True)
    metrics = [("peak_kv_mib", "Peak KV cache per query (MiB, log)"),
               ("kv_reads", "Decode KV token-reads per query (log)")]
    color = {"235B-A22B": PALETTE["blue"], "30B-A3B": PALETTE["orange"]}

    for row, (xkey, xlabel) in enumerate(metrics):
        for col, (task, task_label, marker) in enumerate(TASKS):
            ax = axes[row][col]
            for model, _ in MODELS:
                ste = [cells[(model, task, c)] for c, _ in STE]
                cot = [cells[(model, task, c)] for c, _ in COT]
                ax.plot([c[xkey] for c in ste], [c["accuracy"] for c in ste],
                        "--", lw=0.9, color=color[model], marker=marker, ms=3.4,
                        mfc="white", mec=color[model], mew=0.8)
                ax.plot([c[xkey] for c in cot], [c["accuracy"] for c in cot],
                        "-", lw=1.0, color=color[model], marker=marker, ms=3.4,
                        mfc=color[model], mec=color[model], mew=0.6)
            ax.set_xscale("log")
            ax.set_ylim(0, 108)
            ax.set_yticks(range(0, 101, 20))
            ax.grid(axis="y", lw=0.3, alpha=0.5)
            ax.set_xlabel(xlabel)
            ax.set_title(f"({'abcd'[row * 2 + col]}) {task_label}" + ("" if row else ""),
                         fontsize=6.8, loc="left")

    for ax_row in axes:
        ax_row[0].set_ylabel("Accuracy (%)")

    handles = []
    for model, label in MODELS:
        handles.append(axes[0][0].plot([], [], "-", color=color[model], lw=1.2, label=label)[0])
    handles.append(axes[0][0].plot([], [], "-", color="#555555", lw=1.0, marker="o", ms=3.4,
                                   label="CoT cap sweep")[0])
    handles.append(axes[0][0].plot([], [], "--", color="#555555", lw=0.9, marker="o", ms=3.4,
                                   mfc="white", label="STE length sweep")[0])
    for task, label, marker in TASKS:
        handles.append(axes[0][0].plot([], [], marker=marker, ls="none", ms=4,
                                       mfc="#555555", mec="#555555", label=label)[0])
    axes[0][0].legend(handles=handles, ncol=1, fontsize=5.4, frameon=False,
                      loc="upper left", handlelength=1.8, labelspacing=0.3,
                      bbox_to_anchor=(-0.02, 1.02))
    save(fig, "recovery-pareto")


def sensitivity_figure(data, cells):
    fig, axes = figure_grid(nrows=1, ncols=3, width=WIDE, ratio=0.40, sharey=True)

    # (a) STE filler length
    ax = axes[0]
    for model, _ in MODELS:
        for task, _, marker in TASKS:
            xs = [l for _, l in STE]
            ys = [cells[(model, task, c)]["accuracy"] for c, _ in STE]
            errs = [cells[(model, task, c)]["ci95"] for c, _ in STE]
            color = PALETTE["blue"] if model == "235B-A22B" else PALETTE["orange"]
            ax.errorbar(xs, ys, yerr=errs, marker=marker, ms=3.4, lw=0.9, capsize=1.8,
                        elinewidth=0.6, color=color,
                        mfc=color if marker == "o" else "white", mec=color, mew=0.7)
    ax.set_xticks([l for _, l in STE], [str(l) for _, l in STE])
    ax.set_xlabel("STE filler length (tokens)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("(a) filler length", fontsize=7, loc="left")

    # (b) CoT cap, log x, with realized-token position marked
    ax = axes[1]
    for model, _ in MODELS:
        for task, _, marker in TASKS:
            xs = [l for _, l in COT]
            ys = [cells[(model, task, c)]["accuracy"] for c, _ in COT]
            errs = [cells[(model, task, c)]["ci95"] for c, _ in COT]
            color = PALETTE["blue"] if model == "235B-A22B" else PALETTE["orange"]
            ax.errorbar(xs, ys, yerr=errs, marker=marker, ms=3.4, lw=0.9, capsize=1.8,
                        elinewidth=0.6, color=color,
                        mfc=color if marker == "o" else "white", mec=color, mew=0.7)
    ax.set_xscale("log")
    ax.set_xticks([l for _, l in COT], ["128", "256", "512", "2048"])
    # log minor labels collide with the explicit cap labels
    ax.tick_params(axis="x", which="minor", labelbottom=False, length=2)
    ax.set_xlabel("CoT token cap")
    ax.set_title("(b) chain-of-thought budget", fontsize=7, loc="left")
    ax.axvspan(128, 256, color=PALETTE["red"], alpha=0.10, lw=0)
    ax.annotate("no answer\nchannel", xy=(181, 55), ha="center", va="center",
                fontsize=5.8, color=PALETTE["red"])

    # (c) model size at each fixed STE length
    ax = axes[2]
    for task, _, marker in TASKS:
        for cond, length in STE:
            ys = [cells[(m, task, cond)]["accuracy"] for m, _ in MODELS]
            errs = [cells[(m, task, cond)]["ci95"] for m, _ in MODELS]
            ax.errorbar([0, 1], ys, yerr=errs, marker=marker, ms=3.2, lw=0.8, capsize=1.6,
                        elinewidth=0.5, color=BASELINE if length == 0 else PALETTE["green"],
                        alpha=0.45 + 0.55 * (length / 500),
                        mfc=BASELINE if length == 0 else PALETTE["green"],
                        mec="black", mew=0.3)
    ax.set_xticks([0, 1], ["235B", "30B"])
    ax.set_xlim(-0.4, 1.4)
    ax.set_xlabel("model (total parameters)")
    ax.set_title("(c) model size", fontsize=7, loc="left")

    for ax in axes:
        ax.set_ylim(0, 108)
        ax.set_yticks(range(0, 101, 20))
        ax.grid(axis="y", lw=0.3, alpha=0.5)

    handles = []
    for model, label in MODELS:
        color = PALETTE["blue"] if model == "235B-A22B" else PALETTE["orange"]
        handles.append(axes[0].plot([], [], "-", color=color, lw=0.9, label=label)[0])
    for task, label, marker in TASKS:
        handles.append(axes[0].plot([], [], marker=marker, ls="none", ms=4,
                                    mfc="#555555", mec="#555555", label=label)[0])
    axes[1].legend(handles=handles, ncol=1, fontsize=5.6, frameon=False,
                   loc="lower right", handlelength=1.4, labelspacing=0.3)
    save(fig, "recovery-sensitivity")


def main():
    use_style()
    data, cells = load()
    pareto_figure(data, cells)
    sensitivity_figure(data, cells)


if __name__ == "__main__":
    main()
