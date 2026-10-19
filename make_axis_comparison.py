"""Side-by-side axis comparison: SSBP1 / SERPINF1 / DEPTOR correlation with
deconvolved cell fractions and hypoxia score (TCGA-HNSC).
Input  -> data/results_gene/fraction_corr.csv
Output -> data/results_gene/axis_comparison.png
"""
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

c = pd.read_csv("data/results_gene/fraction_corr.csv")
c = c[~c.fraction.str.contains("|", regex=False)]        # overall, not HPV-stratified

labels = {"prop_Malignant": "Malignant\nfraction",
          "prop_Immune": "Immune\nfraction",
          "prop_Myeloid": "Myeloid\nfraction",
          "prop_NK": "NK\nfraction",
          "Hypoxia_score": "Hypoxia\nscore"}
order = list(labels)
genes = ["SSBP1", "SERPINF1", "DEPTOR"]
cols = {"SSBP1": "#8E44AD", "SERPINF1": "#1E8449", "DEPTOR": "#2E86AB"}

fig, ax = plt.subplots(figsize=(8.8, 4.6))
fig.subplots_adjust(left=0.10, right=0.975, top=0.80, bottom=0.19)
x = np.arange(len(order))
w = 0.26
for i, g in enumerate(genes):
    sub = c[c.gene == g].set_index("fraction").reindex(order)
    off = (i - 1) * w
    bars = ax.bar(x + off, sub.rho.values, w,
                  color=cols[g], edgecolor="#222", lw=0.5, label=g, alpha=0.9)
    for b, r, p in zip(bars, sub.rho.values, sub.p.values):
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        if star:
            ax.text(b.get_x() + b.get_width() / 2,
                    r + (0.016 if r >= 0 else -0.016),
                    star, ha="center", va="bottom" if r >= 0 else "top",
                    fontsize=8.5, color="#111")

ax.axhline(0, color="#333", lw=0.9)
ax.set_xticks(x); ax.set_xticklabels([labels[o] for o in order], fontsize=8.8)
ax.set_ylabel("Spearman ρ with gene expression", fontsize=9.5)
ax.set_ylim(-0.54, 0.34)
ax.grid(axis="y", color="#DDDDDD", lw=0.6)
ax.set_axisbelow(True)
ax.legend(fontsize=8.8, frameon=False, ncol=3, loc="upper center",
          bbox_to_anchor=(0.5, 1.14))
ax.set_title("SSBP1, SERPINF1 and DEPTOR track opposite microenvironmental axes\n"
             "(TCGA-HNSC, n = 487)",
             fontsize=10.6, fontweight="bold", pad=26)
ax.text(0.5, -0.30,
        "* P<0.05   ** P<0.01   *** P<0.001 (Spearman, two-sided).\n"
        "SSBP1: malignant / hypoxic axis.   SERPINF1 and DEPTOR: immune-hot, low-malignant, low-hypoxia axis.",
        transform=ax.transAxes, ha="center", va="top", fontsize=8.0, color="#555")
fig.savefig("data/results_gene/axis_comparison.png", dpi=300, bbox_inches="tight")
print("saved axis_comparison.png")
