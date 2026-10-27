#!/usr/bin/env python3
"""Render the three-panel failure-mode schematic used in the manuscripts.

The figure is conceptual, not data-driven: it uses fixed synthetic score
distributions to illustrate the operational distinction between threshold
drift, ranking collapse, and orientation inversion under a transferred
source-selected threshold. Writes PNG and PDF copies into both the journal
and ACSAC figure directories so either build path can include it.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _panel(ax, benign_scores, attack_scores, threshold, title, subtitle, color_benign, color_attack):
    bins = np.linspace(0.0, 1.0, 41)
    ax.hist(
        benign_scores,
        bins=bins,
        density=True,
        alpha=0.55,
        color=color_benign,
        edgecolor="white",
        linewidth=0.4,
        label="benign",
    )
    ax.hist(
        attack_scores,
        bins=bins,
        density=True,
        alpha=0.55,
        color=color_attack,
        edgecolor="white",
        linewidth=0.4,
        label="attack",
    )
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.3, label="frozen threshold")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 6.0)
    ax.set_xlabel("source detector score")
    ax.set_ylabel("density")
    ax.set_title(title, fontsize=11, pad=8)
    ax.text(
        0.02,
        0.96,
        subtitle,
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "0.8", "boxstyle": "round,pad=0.3"},
    )
    ax.tick_params(axis="both", labelsize=8)


def build_figure(rng: np.random.Generator):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1), sharey=True)

    benign = np.clip(rng.normal(0.20, 0.07, size=4000), 0.0, 1.0)
    attack = np.clip(rng.normal(0.75, 0.08, size=4000), 0.0, 1.0)
    _panel(
        axes[0],
        benign,
        attack,
        threshold=0.38,
        title="(a) Threshold drift",
        subtitle="ranking preserved;\nfrozen cutoff off-budget on target",
        color_benign="#4C78A8",
        color_attack="#E45756",
    )

    benign = np.clip(rng.normal(0.45, 0.14, size=4000), 0.0, 1.0)
    attack = np.clip(rng.normal(0.55, 0.14, size=4000), 0.0, 1.0)
    _panel(
        axes[1],
        benign,
        attack,
        threshold=0.68,
        title="(b) Ranking collapse",
        subtitle="benign/attack overlap;\nno threshold restores detection",
        color_benign="#4C78A8",
        color_attack="#E45756",
    )

    benign = np.clip(rng.normal(0.70, 0.09, size=4000), 0.0, 1.0)
    attack = np.clip(rng.normal(0.30, 0.09, size=4000), 0.0, 1.0)
    _panel(
        axes[2],
        benign,
        attack,
        threshold=0.55,
        title="(c) Orientation inversion",
        subtitle="score direction reversed;\nattacks score below benign",
        color_benign="#4C78A8",
        color_attack="#E45756",
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--journal-figures",
        default="paper/figures",
        help="Directory for the journal manuscript figures.",
    )
    parser.add_argument(
        "--acsac-figures",
        default="paper/acsac/figures",
        help="Directory for the ACSAC manuscript figures.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260413,
        help="Random seed for the synthetic illustration.",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    fig = build_figure(rng)

    for target_dir in (args.journal_figures, args.acsac_figures):
        out_dir = Path(target_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "failure_mode_schematic.png", dpi=220, bbox_inches="tight")
        fig.savefig(out_dir / "failure_mode_schematic.pdf", bbox_inches="tight")

    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
