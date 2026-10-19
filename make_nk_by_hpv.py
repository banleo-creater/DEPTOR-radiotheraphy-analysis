"""Standalone figure: NK cell proportion by HPV status.
A = Xenium single-cell spatial (per-section frac_NK); B = TCGA bulk
deconvolution (EPIC/quanTIseq prop_NK). Saves Figure_NK_by_HPV.png."""
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

C_HPVP = "#E8743B"   # HPV+
C_HPVN = "#3B6EA5"   # HPV-
GRID   = "#DDDDDD"

x = pd.read_csv("data/results_xenium/xenium_summary.csv")
x["hpv"] = x["hpv"].str.replace("HPV", "", regex=False).str.strip()
xen_p = x.loc[x.hpv == "+", "frac_NK"].values
xen_n = x.loc[x.hpv == "-", "frac_NK"].values

d = pd.read_csv("data/results_deconv/deconv_per_sample.csv")
tcga_p = d.loc[d.HPV == "HPV+", "prop_NK"].dropna().values
tcga_n = d.loc[d.HPV == "HPV-", "prop_NK"].dropna().values

def welsh(a, b):
    return stats.ttest_ind(a, b, equal_var=False).pvalue

pv_x = welsh(xen_p, xen_n)
pv_t = welsh(tcga_p, tcga_n)

fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.4))
fig.subplots_adjust(left=0.09, right=0.97, top=0.80, bottom=0.13, wspace=0.30)

# ---------- Panel A: Xenium spatial ----------
ax = axes[0]
data_a = [xen_p * 100, xen_n * 100]
bp = ax.boxplot(data_a, positions=[1, 2], widths=0.5, patch_artist=True,
                showfliers=False, medianprops=dict(color="#222", lw=1.4),
                whiskerprops=dict(color="#444"), capprops=dict(color="#444"))
for patch, col in zip(bp["boxes"], [C_HPVP, C_HPVN]):
    patch.set_facecolor(col); patch.set_alpha(0.55)
rng = np.random.default_rng(7)
for i, arr in enumerate(data_a, start=1):
    ax.scatter(rng.normal(i, 0.04, len(arr)), arr, s=34, color=[C_HPVP, C_HPVN][i-1],
               edgecolor="white", linewidth=0.7, zorder=3)
ax.set_xticks([1, 2]); ax.set_xticklabels([f"HPV+\n(n={len(xen_p)})", f"HPV−\n(n={len(xen_n)})"], fontsize=9)
ax.set_ylabel("NK fraction of all cells (%)", fontsize=9.5)
ax.set_title("A. Xenium spatial (single-cell resolved)", fontsize=10, fontweight="semibold")
ax.grid(axis="y", color=GRID, lw=0.7)
ax.set_ylim(0, max(xen_p.max(), xen_n.max()) * 100 * 1.22)
# annotation: means above each box + p inside axes
mp, mn = xen_p.mean()*100, xen_n.mean()*100
ax.text(0.25, 0.88, f"mean {mp:.1f}%", transform=ax.transAxes, ha="center", va="top",
        fontsize=8.2, color=C_HPVP, fontweight="bold")
ax.text(0.75, 0.88, f"mean {mn:.1f}%", transform=ax.transAxes, ha="center", va="top",
        fontsize=8.2, color=C_HPVN, fontweight="bold")
ax.text(0.5, 0.965, f"Welch p = {pv_x:.2f}  (n.s.)",
        transform=ax.transAxes, ha="center", va="top", fontsize=8.4, style="italic", color="#555")

# ---------- Panel B: TCGA bulk deconvolution ----------
ax = axes[1]
data_b = [tcga_p * 100, tcga_n * 100]
bp = ax.boxplot(data_b, positions=[1, 2], widths=0.5, patch_artist=True,
                showfliers=False, medianprops=dict(color="#222", lw=1.4),
                whiskerprops=dict(color="#444"), capprops=dict(color="#444"))
for patch, col in zip(bp["boxes"], [C_HPVP, C_HPVN]):
    patch.set_facecolor(col); patch.set_alpha(0.55)
rng = np.random.default_rng(3)
for i, arr in enumerate(data_b, start=1):
    ax.scatter(rng.normal(i, 0.04, len(arr)), arr, s=9,
               color=[C_HPVP, C_HPVN][i-1], edgecolor="none", alpha=0.45, zorder=3)
ax.set_xticks([1, 2]); ax.set_xticklabels([f"HPV+\n(n={len(tcga_p)})", f"HPV−\n(n={len(tcga_n)})"], fontsize=9)
ax.set_ylabel("NK fraction (deconvolved, %)", fontsize=9.5)
ax.set_title("B. TCGA-HNSC bulk deconvolution", fontsize=10, fontweight="semibold")
ax.grid(axis="y", color=GRID, lw=0.7)
ax.set_ylim(0, max(tcga_p.max(), tcga_n.max()) * 100 * 1.14)
mp, mn = tcga_p.mean()*100, tcga_n.mean()*100
ax.text(0.25, 0.88, f"mean {mp:.1f}%", transform=ax.transAxes, ha="center", va="top",
        fontsize=8.2, color=C_HPVP, fontweight="bold")
ax.text(0.75, 0.88, f"mean {mn:.1f}%", transform=ax.transAxes, ha="center", va="top",
        fontsize=8.2, color=C_HPVN, fontweight="bold")
ax.text(0.5, 0.965, "Welch p < 0.001",
        transform=ax.transAxes, ha="center", va="top", fontsize=8.4, style="italic", color="#555")

fig.suptitle("NK cell proportion is modestly higher in HPV-positive HNSCC",
             fontsize=11.5, fontweight="bold", y=0.96)

fig.savefig("data/results_ms_figures/Figure_NK_by_HPV.png", dpi=300, bbox_inches="tight")
print("saved Figure_NK_by_HPV.png")
print(f"Xenium HPV+ {xen_p.mean():.4f} (n={len(xen_p)}), HPV- {xen_n.mean():.4f} (n={len(xen_n)}), p={pv_x:.3f}")
print(f"TCGA   HPV+ {tcga_p.mean():.4f} (n={len(tcga_p)}), HPV- {tcga_n.mean():.4f} (n={len(tcga_n)}), p={pv_t:.3e}")
