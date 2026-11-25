"""Accuracy versus decode-stage KV-cache traffic, STE filler vs full CoT.

Regenerate:  uv run --no-project --with matplotlib --with numpy python kv_tradeoff.py
"""
import json
import os

import numpy as np
from orx_figstyle import BASELINE, PALETTE, WIDE, family, figure, save, use_style

HERE = os.path.dirname(os.path.abspath(__file__))
TASKS = ["multiplication", "arithmetic"]
LABELS = {"multiplication": "4-digit multiplication", "arithmetic": "multi-step arithmetic"}
MARKERS = {"multiplication": "o", "arithmetic": "s"}


def load():
    with open(os.path.join(HERE, "kv_efficiency.json")) as handle:
        return json.load(handle)


def main():
    use_style()
    data = load()

    conditions = [
        ("Baseline (no CoT)", "baseline", BASELINE),
        ("STE filler", "ste_filler", family("orange", 1)[0]),
        ("Full CoT", "cot", family("blue", 1)[0]),
    ]

    fig, ax = figure(width=WIDE, ratio=0.46)
    for task in TASKS:
        for label, key, color in conditions:
            cell = data[f"{task}/{key}"]
            ax.scatter(cell["reads"], cell["acc"], s=42, marker=MARKERS[task],
                       color=color, edgecolor="black", linewidth=0.4, zorder=3)
            ax.annotate(f"{cell['acc']:.1f}%", (cell["reads"], cell["acc"]),
                        textcoords="offset points", xytext=(7, 5), fontsize=6.0,
                        color="#222222")

        cot = data[f"{task}/cot"]
        ste = data[f"{task}/ste_filler"]
        ax.annotate("", xy=(ste["reads"], ste["acc"]), xytext=(cot["reads"], cot["acc"]),
                    arrowprops={"arrowstyle": "-|>", "lw": 0.9, "color": PALETTE["red"],
                                "shrinkA": 6, "shrinkB": 6, "alpha": 0.75})
        ax.annotate(
            f"{cot['reads'] / ste['reads']:.0f}× fewer reads\n"
            f"{ste['acc'] - cot['acc']:+.1f} pp",
            xy=(np.sqrt(ste["reads"] * cot["reads"]),
                ste["acc"] + 0.55 * (cot["acc"] - ste["acc"])),
            ha="center", va="center", fontsize=6.2, color=PALETTE["red"],
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2},
            zorder=6,
        )

    ax.set_xscale("log")
    ax.set_xlabel("KV-cache token-reads per query (decode, log scale)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 108)
    ax.set_yticks(range(0, 101, 20))

    handles = [ax.scatter([], [], s=34, color=c, edgecolor="black", linewidth=0.4, label=l)
               for l, _, c in conditions]
    handles += [ax.scatter([], [], s=34, color="white", edgecolor="black", linewidth=0.6,
                           marker=MARKERS[t], label=LABELS[t]) for t in TASKS]
    ax.legend(handles=handles, ncol=5, loc="upper center", fontsize=5.8, frameon=False,
              bbox_to_anchor=(0.5, 1.06), columnspacing=1.0, handletextpad=0.4)

    save(fig, "kv-tradeoff")


if __name__ == "__main__":
    main()
