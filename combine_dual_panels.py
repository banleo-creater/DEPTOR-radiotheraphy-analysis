#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Combine the dual-axis scatter panels into one publication-style figure."""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from matplotlib.lines import Line2D

BASE = "WorkBuddy/2026-10-17-11-15-46"
OUT  = os.path.join(BASE, "data", "results_dual")
df = pd.read_csv(os.path.join(OUT, "dual_axis_summary.csv"))

def fitline(ax, x, y, color):
    if len(x) >= 3:
        b, a = np.polyfit(x, y, 1)
        xs = np.linspace(np.min(x), np.max(x), 50)
        ax.plot(xs, a + b * xs, "--", color=color, lw=1.4, alpha=0.8)

fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))

# A: tumor DEPTOR vs myeloid DEPTOR
ax = axes[0]
s = df[["DEPTOR_malignant_mean","DEPTOR_myeloid_mean"]].dropna()
rho, p = spearmanr(s.DEPTOR_malignant_mean, s.DEPTOR_myeloid_mean)
ax.scatter(s.DEPTOR_malignant_mean, s.DEPTOR_myeloid_mean, c="#2ca02c", s=70, edgecolor="k")
for _, r in df.iterrows():
    if pd.notna(r.DEPTOR_malignant_mean) and pd.notna(r.DEPTOR_myeloid_mean):
        ax.annotate(r.patient, (r.DEPTOR_malignant_mean, r.DEPTOR_myeloid_mean),
                    fontsize=7, xytext=(3,3), textcoords="offset points")
fitline(ax, s.DEPTOR_malignant_mean.values, s.DEPTOR_myeloid_mean.values, "#2ca02c")
ax.set_xlabel("DEPTOR in malignant (tumor) cells"); ax.set_ylabel("DEPTOR in myeloid (immune) cells")
ax.set_title(f"A  Tumor vs myeloid DEPTOR\nSpearman rho={rho:+.2f}, p={p:.2g}, n={len(s)}")
ax.text(0.02, 0.02, "no anti-correlation\n(independent axes)", transform=ax.transAxes,
        fontsize=8, va="bottom", color="#444")

# B: myeloid DEPTOR vs NK fraction
ax = axes[1]
s2 = df[["DEPTOR_myeloid_mean","NK_fraction_of_immune"]].dropna()
rho, p = spearmanr(s2.DEPTOR_myeloid_mean, s2.NK_fraction_of_immune)
ax.scatter(s2.DEPTOR_myeloid_mean, s2.NK_fraction_of_immune, c="#1f77b4", s=70, edgecolor="k")
for _, r in df.iterrows():
    if pd.notna(r.DEPTOR_myeloid_mean) and pd.notna(r.NK_fraction_of_immune):
        ax.annotate(r.patient, (r.DEPTOR_myeloid_mean, r.NK_fraction_of_immune),
                    fontsize=7, xytext=(3,3), textcoords="offset points")
fitline(ax, s2.DEPTOR_myeloid_mean.values, s2.NK_fraction_of_immune.values, "#1f77b4")
ax.set_xlabel("DEPTOR in myeloid (immune) cells"); ax.set_ylabel("NK fraction of immune")
ax.set_title(f"B  Myeloid DEPTOR vs NK\nSpearman rho={rho:+.2f}, p={p:.3g}, n={len(s2)}")

# C: tumor DEPTOR vs NK fraction
ax = axes[2]
s3 = df[["DEPTOR_malignant_mean","NK_fraction_of_immune"]].dropna()
rho, p = spearmanr(s3.DEPTOR_malignant_mean, s3.NK_fraction_of_immune)
ax.scatter(s3.DEPTOR_malignant_mean, s3.NK_fraction_of_immune, c="#9467bd", s=70, edgecolor="k")
for _, r in df.iterrows():
    if pd.notna(r.DEPTOR_malignant_mean) and pd.notna(r.NK_fraction_of_immune):
        ax.annotate(r.patient, (r.DEPTOR_malignant_mean, r.NK_fraction_of_immune),
                    fontsize=7, xytext=(3,3), textcoords="offset points")
fitline(ax, s3.DEPTOR_malignant_mean.values, s3.NK_fraction_of_immune.values, "#9467bd")
ax.set_xlabel("DEPTOR in malignant (tumor) cells"); ax.set_ylabel("NK fraction of immune")
ax.set_title(f"C  Tumor DEPTOR vs NK\nSpearman rho={rho:+.2f}, p={p:.3g}, n={len(s3)}")

plt.suptitle("GSE164690 dual-axis: DEPTOR in tumor vs immune compartments and NK abundance",
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "dual_axis_SUMMARY_panels.png"), dpi=180, bbox_inches="tight")
plt.close()
print("saved dual_axis_SUMMARY_panels.png")
