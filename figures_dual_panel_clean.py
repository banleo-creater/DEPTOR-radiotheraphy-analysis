#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Clean three-panel dual-axis figure (no overlapping patient labels).
Only influential points are annotated; all points remain visible.

Output -> data/results_final_figures/F5_dual_axis_panels.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
import warnings
warnings.filterwarnings("ignore")

BASE = "WorkBuddy/2026-10-17-11-15-46"
OUT = os.path.join(BASE, "data", "results_final_figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9.5,
    "axes.titlesize": 9.8, "axes.labelsize": 9.2,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.linewidth": 0.9, "figure.dpi": 110,
    "savefig.dpi": 400, "savefig.bbox": "tight", "savefig.pad_inches": 0.1,
})

d = pd.read_csv(os.path.join(BASE, "data", "results_dual", "dual_axis_summary.csv"))
d = d.rename(columns={"DEPTOR_malignant_mean": "mal", "DEPTOR_myeloid_mean": "mye",
                      "NK_fraction_of_immune": "nk", "DEPTOR_NK_mean": "nkexpr"})

C_PT, C_FIT, C_LAB = "#3b4cc0", "#8c8c8c", "#222222"


def annotate_point(ax, x, y, name, dx, dy, ha="left"):
    """Deterministic annotation with an opaque background (no collisions)."""
    ax.annotate(name, xy=(x, y), xytext=(x + dx, y + dy),
                fontsize=8.2, color=C_LAB, ha=ha, va="center", zorder=6,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85),
                arrowprops=dict(arrowstyle="-", lw=0.6, color="0.5", shrinkA=3, shrinkB=3))


def panel(ax, x, y, names, xlabel, ylabel, title, note=None):
    ax.scatter(x, y, s=42, facecolor=C_PT, edgecolor="white", linewidth=0.8,
               alpha=0.92, zorder=4)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() > 2:
        b, a = np.polyfit(x[ok], y[ok], 1)
        xs = np.linspace(np.nanmin(x[ok]), np.nanmax(x[ok]), 50)
        ax.plot(xs, a + b * xs, ls="--", lw=1.1, color=C_FIT, zorder=2)
    rho, p = spearmanr(x[ok], y[ok])
    ax.set_title(f"{title}\nSpearman ρ = {rho:+.2f}, P = {p:.3g}, n = {int(ok.sum())}",
                 loc="left", fontsize=9.4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    if note:
        ax.text(0.03, 0.04, note, transform=ax.transAxes, fontsize=7.8, color="0.4",
                va="bottom", ha="left")
    return rho, p


fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.5))
fig.subplots_adjust(wspace=0.30, left=0.055, right=0.99, top=0.72, bottom=0.155)

# A: malignant vs myeloid DEPTOR
x, y = d.mal.values, d.mye.values
panel(axes[0], x, y, d.patient.values,
      "DEPTOR in malignant cells (mean log1p)",
      "DEPTOR in myeloid cells (mean log1p)",
      "A  Malignant versus myeloid DEPTOR")
i07 = int(np.where(d.patient.values == "HN07")[0][0])
xr = np.nanmax(x) - np.nanmin(x)
yr = np.nanmax(y) - np.nanmin(y)
annotate_point(axes[0], x[i07], y[i07], "HN07", dx=-0.030 * xr, dy=0.11 * yr, ha="right")
axes[0].text(0.40, 0.04, "each dot = one patient",
             transform=axes[0].transAxes, fontsize=7.6, color="0.45", va="bottom",
             bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))

# B: myeloid DEPTOR vs NK fraction
x, y = d.mye.values, d.nk.values * 100
panel(axes[1], x, y, d.patient.values,
      "DEPTOR in myeloid cells (mean log1p)",
      "NK cells (% of immune infiltrate)",
      "B  Myeloid DEPTOR and NK abundance")
axes[1].text(0.40, 0.04, "each dot = one patient", transform=axes[1].transAxes,
             fontsize=7.6, color="0.45", va="bottom",
             bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))

# C: malignant DEPTOR vs NK fraction
x, y = d.mal.values, d.nk.values * 100
panel(axes[2], x, y, d.patient.values,
      "DEPTOR in malignant cells (mean log1p)",
      "NK cells (% of immune infiltrate)",
      "C  Malignant DEPTOR and NK abundance")
annotate_point(axes[2], x[i07], y[i07], "HN07",
               dx=-0.030 * (np.nanmax(x) - np.nanmin(x)),
               dy=-0.10 * (np.nanmax(y) - np.nanmin(y)), ha="right")
axes[2].text(0.40, 0.04, "each dot = one patient", transform=axes[2].transAxes,
             fontsize=7.6, color="0.45", va="bottom",
             bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))

fig.suptitle("DEPTOR in two compartments: independent axes that both track NK scarcity (GSE164690)",
             fontsize=10.8, fontweight="bold", y=0.965)
fig.savefig(os.path.join(OUT, "F5_dual_axis_panels.png"))
plt.close(fig)
print("saved: F5_dual_axis_panels.png")
