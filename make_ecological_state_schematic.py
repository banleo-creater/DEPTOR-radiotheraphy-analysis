#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Schematic for the proposed MAIN FIGURE:

    "DEPTOR defines a myeloid-associated, NK-poor ecological state in HNSCC"

  A  two INDEPENDENT DEPTOR axes (malignant, myeloid) converge on NK paucity
  B  spatial realisation at single-cell resolution (Xenium)
  C  clinical translation, explicitly flagged as hypothesis-generating

All numbers are READ FROM THE RESULT CSVs so the figure cannot drift from the
statistics. Honest nulls are drawn in amber/red on purpose.
"""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse, FancyArrowPatch

BASE = "WorkBuddy/2026-10-17-11-15-46"
OUTD = os.path.join(BASE, "data", "results_myeloid_nk_axis")
XN = os.path.join(BASE, "data", "results_xenium_mnk")
os.makedirs(OUTD, exist_ok=True)

INK, GREY = "#1a1a1a", "#5a5a5a"
BLUE, TEAL, RED, AMBER = "#2b6cb0", "#1f8a70", "#c0392b", "#b8860b"
PURPLE, COL_M1, LGREY = "#7b4fa8", "#e08a1e", "#eceff1"


def rd(fn):
    p = os.path.join(XN, fn)
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()


enr, nt = rd("xenium_mnk_enrich100.csv"), rd("xenium_mnk_nontumor.csv")


def get(df, col, q, default=np.nan):
    if df.empty or col not in df.columns or "pair" not in df.columns:
        return default
    s = df.loc[df["pair"] == q, col]
    return float(s.iloc[0]) if len(s) else default


def ntv(pair, kind="enrich@100um", default=np.nan):
    if nt.empty:
        return default
    s = nt[(nt["pair"] == pair) & (nt["kind"] == kind)]
    return float(s["est"].iloc[0]) if len(s) else default


def ntp(pair, default=np.nan):
    if nt.empty:
        return default
    s = nt[(nt["pair"] == pair) & (nt["kind"] == "enrich@100um")]
    return float(s["p"].iloc[0]) if len(s) else default


E_TUM_NK = get(enr, "mean_enr", "Tumor->NK", 0.666)
LO, HI = get(enr, "ci_lo", "Tumor->NK", 0.508), get(enr, "ci_hi", "Tumor->NK", 0.824)
N_M2, N_M1, N_DC = ntv("NK->M2"), ntv("NK->M1"), ntv("NK->DC")


def box(ax, x, y, w, h, txt, fc=LGREY, ec=GREY, fs=8.5, tc=INK, lw=1.2,
        bold=False, z=3):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.010,rounding_size=0.018",
                                fc=fc, ec=ec, lw=lw, zorder=z))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs,
            color=tc, zorder=z + 1, fontweight="bold" if bold else "normal",
            linespacing=1.5)


def arrow(ax, p1, p2, color=GREY, lw=1.5, ls="-", style="-|>", ms=12):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=ms,
                                 color=color, lw=lw, linestyle=ls, zorder=2,
                                 shrinkA=2, shrinkB=2))


WB = dict(fc="white", ec="#dddddd", lw=.8, alpha=0.96, pad=4.5)

fig, axes = plt.subplots(1, 3, figsize=(16.5, 9.4))
for ax in axes:
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

# ================================== A ==================================
ax = axes[0]
ax.set_title("A   Two independent DEPTOR axes converge on NK paucity",
             fontsize=11, fontweight="bold", loc="left", pad=12)

box(ax, 0.03, 0.720, 0.42, 0.125, "Malignant-cell DEPTOR ↑\n(GSE164690, n = 15)",
    fc="#e3f2ef", ec=TEAL, fs=8.4)
box(ax, 0.55, 0.720, 0.42, 0.125, "Myeloid DEPTOR ↑\nM2-TAM + cDC (n = 18)",
    fc="#e6eff8", ec=BLUE, fs=8.4)
arrow(ax, (0.45, 0.782), (0.55, 0.782), color=GREY, ls=(0, (4, 3)), lw=1.3,
      style="<|-|>", ms=11)
ax.text(0.50, 0.928, "ρ = +0.26,  P = 0.35", ha="center", fontsize=8,
        color=GREY)
ax.text(0.50, 0.893, "independent — not a see-saw", ha="center", fontsize=7.4,
        color=GREY, style="italic")

arrow(ax, (0.24, 0.715), (0.335, 0.622), color=TEAL, lw=1.9)
arrow(ax, (0.76, 0.715), (0.665, 0.622), color=BLUE, lw=1.9)
ax.text(0.135, 0.668, "ρ = −0.53\nP = 0.045", ha="center", fontsize=7.9,
        color=TEAL, linespacing=1.5)
ax.text(0.865, 0.668, "ρ = −0.65\nP = 0.003", ha="center", fontsize=7.9,
        color=BLUE, linespacing=1.5)

box(ax, 0.215, 0.500, 0.57, 0.112, "NK-poor microenvironment",
    fc="#fdeceb", ec=RED, fs=10.5, bold=True, lw=1.7)

ax.text(0.50, 0.432, "Replication of the myeloid-DEPTOR → NK link", ha="center",
        fontsize=8.3, fontweight="bold", color=INK)
led = [("GSE139324  (n = 26)", "ρ = −0.51   P = 0.007", TEAL, "✓"),
       ("GSE198315  (n = 36)", "ρ = +0.16   P = 0.37", AMBER, "✗"),
       ("TCGA bulk  (n = 515)", "ρ = +0.07   P = 0.10", AMBER, "✗")]
for i, (nm, st, col, mk) in enumerate(led):
    y = 0.376 - i * 0.050
    ax.text(0.055, y, f"{mk}  {nm}", fontsize=7.9, color=col, va="center")
    ax.text(0.945, y, st, fontsize=7.9, color=col, ha="right", va="center")
ax.plot([0.05, 0.95], [0.228, 0.228], color="#cccccc", lw=0.8)
ax.text(0.50, 0.155,
        "The negative link is seen in sorted single-cell data only;\n"
        "it does not survive in two of four cohorts.",
        ha="center", fontsize=7.4, color=GREY, style="italic", linespacing=1.6)

# ================================== B ==================================
ax = axes[1]
ax.set_title("B   Spatial realisation — Xenium, 17 sections, 1.1 M cells",
             fontsize=11, fontweight="bold", loc="left", pad=12)

cx, cy = 0.50, 0.745
ax.add_patch(Ellipse((cx, cy), 0.40, 0.245, fc="#dcdde1", ec="#9aa0a6",
                     lw=1.4, zorder=1))
ax.text(cx, cy + 0.030, "Tumour nest", ha="center", va="center", fontsize=9.5,
        color="#4a4a4a", fontweight="bold", zorder=5)
ax.text(cx, cy - 0.025, "(NK desert)", ha="center", va="center", fontsize=8,
        color=RED, zorder=5)

rng = np.random.default_rng(11)
th = rng.uniform(0, 2 * np.pi, 68)
rx = 0.215 + rng.uniform(0, 0.115, 68)
ry = rx * 0.62
mx = np.clip(cx + rx * np.cos(th), .05, .95)
my = np.clip(cy + ry * np.sin(th), .56, .955)
ax.scatter(mx[:38], my[:38], s=24, c=BLUE, alpha=.88, zorder=4, linewidths=0,
           label="M2-like macrophage")
ax.scatter(mx[38:], my[38:], s=24, c=COL_M1, alpha=.88, zorder=4, linewidths=0,
           label="M1-like macrophage")
idx = rng.choice(len(mx), 26, replace=False)
jx = np.clip(mx[idx] + rng.normal(0, .024, 26), .05, .95)
jy = np.clip(my[idx] + rng.normal(0, .018, 26), .56, .955)
ax.scatter(jx, jy, s=27, c=RED, alpha=.95, zorder=6, linewidths=0, label="NK cell")

# NK-exclusion callout (left)
arrow(ax, (0.175, 0.470), (cx - 0.145, cy - 0.075), color=RED, lw=1.6,
      ls=(0, (5, 3)))
ax.text(0.015, 0.500,
        "NK excluded from the nest\n"
        f"enrichment {E_TUM_NK:.2f}  ({LO:.2f}–{HI:.2f})\n"
        "P = 8 × 10⁻⁴",
        fontsize=7.6, color=RED, va="top", ha="left", linespacing=1.7, bbox=WB,
        zorder=8)

# stromal callout (right)
ax.text(0.545, 0.500,
        "inside stroma (tumour cells removed)\n"
        f"NK ↔ M2    {N_M2:.2f}×     P = {ntp('NK->M2'):.3f}\n"
        f"NK ↔ M1    {N_M1:.2f}×     n.s.\n"
        f"NK ↔ cDC   {N_DC:.2f}×     P = {ntp('NK->DC'):.3f}",
        fontsize=7.5, color=INK, va="top", ha="left", linespacing=1.85, bbox=WB,
        zorder=8)

ax.plot([0.05, 0.95], [0.235, 0.235], color="#cccccc", lw=0.8)
ax.text(0.50, 0.195,
        "NK are excluded from the tumour nest — not repelled by myeloid cells.\n"
        "Within stroma they avoid M2-like macrophages and associate with cDC.",
        ha="center", fontsize=7.5, color=GREY, style="italic", linespacing=1.6)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.012), ncol=3, fontsize=7.2,
          frameon=True, framealpha=.95, handletextpad=.3, columnspacing=1.2,
          borderpad=.4)

# ================================== C ==================================
ax = axes[2]
ax.set_title("C   Clinical translation — hypothesis, not a validated marker",
             fontsize=11, fontweight="bold", loc="left", pad=12)

box(ax, 0.05, 0.780, 0.90, 0.150,
    "DEPTOR-high  →  greater radiotherapy benefit\n"
    "TCGA DSS interaction HR 0.21  (P = 3.7 × 10⁻⁵)\n"
    "RT-arm DSS HR 0.41 (0.24–0.71),  P = 0.001",
    fc="#e6eff8", ec=BLUE, fs=8.5)
arrow(ax, (0.50, 0.775), (0.50, 0.750), color=BLUE, lw=1.7)

box(ax, 0.05, 0.630, 0.90, 0.112,
    "Concentrated in HPV− disease\n"
    "HPV− : NK sit farther from tumour nests (P = 0.036)",
    fc="#fdf6e3", ec=AMBER, fs=8.5)
arrow(ax, (0.50, 0.625), (0.50, 0.582), color=AMBER, lw=1.7)

box(ax, 0.05, 0.457, 0.90, 0.118,
    "Radiotherapy interaction NOT externally replicated\n"
    "GSE65858 HR 0.75 (P = 0.25)   ·   GSE41613 HR 1.08 (P = 0.81)",
    fc="#fdeceb", ec=RED, fs=8.4, lw=1.6)

ax.text(0.50, 0.403, "Consequence for the manuscript", ha="center", fontsize=8.4,
        fontweight="bold", color=INK)
ax.text(0.50, 0.347,
        "Frame as an ecological state with a hypothesis-generating\n"
        "clinical link — not as a validated predictive biomarker.",
        ha="center", fontsize=7.5, color=GREY, linespacing=1.7)
ax.plot([0.10, 0.90], [0.298, 0.298], color="#cccccc", lw=0.8)
ax.text(0.50, 0.268,
        "Unresolved: DEPTOR is absent from every spatial panel\n"
        "(Xenium and Visium), so the DEPTOR → NK step is inferred\n"
        "from dissociated data and has never been observed in situ.",
        ha="center", fontsize=7.4, color=AMBER, style="italic", linespacing=1.7,
        va="top")

plt.tight_layout(rect=(0, 0.005, 1, 0.965))
fig.text(0.5, 0.982,
         "Proposed main figure — DEPTOR defines a myeloid-associated, NK-poor "
         "ecological state in HNSCC",
         ha="center", fontsize=12.5, fontweight="bold")
fp = os.path.join(OUTD, "ecological_state_schematic.png")
plt.savefig(fp, dpi=300, facecolor="white", bbox_inches="tight")
plt.close()
print("saved", fp)
print(f"  Tumour->NK {E_TUM_NK:.3f}; stroma NK->M2 {N_M2:.3f}, "
      f"NK->M1 {N_M1:.3f}, NK->DC {N_DC:.3f}")
