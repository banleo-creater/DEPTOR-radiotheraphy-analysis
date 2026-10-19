#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Why is the DEPTOR -> NK link strong in sorted single cells (rho = -0.68) but
absent in TCGA bulk (rho = +0.07, n = 515)?

Four explanations are TESTED, not assumed:

  E0  IDENTIFIABILITY  — can bulk data even represent the scRNA quantity?
                         (diagnostic: are the compartment DEPTOR splits
                          composition-only, i.e. built from CONSTANT reference
                          weights, or do they carry sample-level expression?)
  E1  DENOMINATOR      — scRNA measured NK as a fraction OF IMMUNE cells;
                         bulk prop_NK is a fraction of ALL cells.
  E2  DILUTION         — exact covariance decomposition:
                         DEPTOR = sum_c contrib_c  =>
                         cov(DEPTOR, NK) = sum_c cov(contrib_c, NK).
                         (valid arithmetic; interpretation caveated by E0)
  E3  STRATUM          — the effect only exists in immune-rich tumours.
  E4  MEASUREMENT      — the bulk NK estimate itself is unreliable.
"""
import os, warnings, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr, chi2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

BASE = "WorkBuddy/2026-10-17-11-15-46"
OUT = os.path.join(BASE, "data", "results_bulk_vs_sc")
os.makedirs(OUT, exist_ok=True)

d = pd.read_csv(os.path.join(BASE, "data", "results_deconv", "deconv_per_sample.csv"))
epic = pd.read_csv(os.path.join(BASE, "data", "results_epic_quantiseq",
                                "TCGA_EPIC_fractions.csv"))
epic["EPIC_immune"] = epic[["Bcells", "CD4_Tcells", "CD8_Tcells",
                            "Macrophages", "NKcells"]].sum(axis=1)
qt = pd.read_csv(os.path.join(BASE, "data", "results_epic_quantiseq",
                              "TCGA_quantiseq_fractions.csv"))
emp = pd.read_csv(os.path.join(BASE, "data", "results_epic_quantiseq",
                               "TCGA_EPIC_DEPTOR_split_empirical.csv"))
d = d.merge(epic[["sampleId", "NKcells", "Macrophages", "otherCells", "CAFs",
                  "EPIC_immune"]], on="sampleId", how="left")
d = d.merge(qt[["sampleId", "NK.cells"]], on="sampleId", how="left")

# ---- deconvolution-free purity / immune proxies straight from expression ----
# These do not depend on any deconvolution model, so they are used as the
# PRIMARY stratifiers; the deconvolution proportions are secondary.
hx = pd.read_csv(os.path.join(BASE, "data", "tcga_hnsc_hiseq.gz"), sep="\t",
                 index_col=0, compression="gzip")
hx = hx[[c for c in hx.columns if c.endswith("-01")]]     # tumours only
prox = pd.DataFrame({"sampleId": list(hx.columns),
                     "CD45_expr": hx.loc["PTPRC"].values,
                     "EPCAM_expr": hx.loc["EPCAM"].values,
                     "KRT_expr": ((hx.loc["KRT5"] + hx.loc["KRT14"]) / 2.0).values})
d = d.merge(prox, on="sampleId", how="left")
print(f"  CD45 / EPCAM matched for {int(d.CD45_expr.notna().sum())} / {len(d)} tumours")

d["NK_of_all"] = d["prop_NK"]
d["NK_of_immune"] = d["prop_NK"] / d["prop_Immune"]

rows = []


def add(part, test, n, est, p=np.nan, note=""):
    rows.append(dict(part=part, test=test, n=n, estimate=est, p=p, note=note))
    ps = "n/a" if (isinstance(p, float) and np.isnan(p)) else f"{p:.3g}"
    print(f"  [{part}] {test:50s} r/rho = {est:+.4f}  p = {ps}  {note}")


def sp(x, y, part, test, note=""):
    m = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(m) < 20:
        return np.nan, np.nan
    r, p = spearmanr(m.x, m.y)
    add(part, test, len(m), r, p, note)
    return r, p


print("=" * 78)
print("TCGA bulk vs sorted single-cell: reconciling DEPTOR -> NK")
print("=" * 78)
print(f"n = {len(d)}")

# ======================= E0  identifiability diagnostic =======================
print("\n--- E0  can bulk data even represent 'myeloid DEPTOR'? ---")
em = d.DEPTOR_myeloid_frac / d.prop_Myeloid      # would be expr_myeloid / total
et = d.DEPTOR_tumor_frac / d.prop_Malignant
es = (1 - d.DEPTOR_tumor_frac - d.DEPTOR_immune_frac) / (d.prop_Stroma + d.prop_Endo)
r_em_et = float(np.corrcoef(em, et)[0, 1])
ratio_sd = float((em / et).std())
add("E0", "corr(expr_myeloid proxy, expr_tumour proxy)", len(d), r_em_et, np.nan,
    "1.000 => the three proxies are the SAME variable")
add("E0", "SD of ratio expr_myeloid/expr_tumour", len(d), ratio_sd, np.nan,
    "0 => separated only by a constant")
b_sd = emp[["beta_Tumor", "beta_Immune", "beta_Other"]].std(ddof=1)
add("E0", "SD of EPIC-empirical beta_Tumor", len(emp), float(b_sd["beta_Tumor"]),
    np.nan, "0 => compartment weight is a CONSTANT")
add("E0", "SD of EPIC-empirical beta_Immune", len(emp), float(b_sd["beta_Immune"]),
    np.nan, "0 => constant")
X = np.column_stack([np.ones(len(d)), d.prop_Malignant, d.prop_Stroma, d.prop_Endo,
                     d.prop_NK, d.prop_Myeloid, d.prop_Tcell, d.prop_Bcell])
bb, *_ = np.linalg.lstsq(X, np.log(d.DEPTOR.values), rcond=None)
r2 = float(1 - ((np.log(d.DEPTOR.values) - X @ bb) ** 2).sum() /
           ((np.log(d.DEPTOR.values) - np.log(d.DEPTOR.values).mean()) ** 2).sum())
add("E0", "R2 of log(DEPTOR) ~ all compartment proportions", len(d), r2, np.nan,
    "measured DEPTOR is NOT composition-only")
print(f"    -> compartment DEPTOR splits use CONSTANT reference weights;")
print(f"       they carry no sample-level compartment expression.")

# ======================= E1  denominator + proxy validation =======================
print("\n--- E1  denominator, NK estimator, and proxy validation ---")
sp(d.DEPTOR, d.NK_of_all, "E1", "DEPTOR vs NK fraction of ALL cells", "published null")
sp(d.DEPTOR, d.NK_of_immune, "E1", "DEPTOR vs NK fraction of IMMUNE cells",
   "scRNA denominator")
sp(d.DEPTOR, d.NKcells, "E1", "DEPTOR vs EPIC NKcells", "")
sp(d.DEPTOR, d["NK.cells"], "E1", "DEPTOR vs quanTIseq NK.cells", "")
print("    proxy validation — which deconvolution tracks CD45 expression?")
sp(d.CD45_expr, d.prop_Immune, "E1", "CD45 vs custom prop_Immune", "")
sp(d.CD45_expr, d.EPIC_immune, "E1", "CD45 vs EPIC immune sum", "")
sp(d.KRT_expr, d.prop_Malignant, "E1", "KRT5/14 vs custom prop_Malignant", "")
sp(d.KRT_expr, d.CAFs, "E1", "KRT5/14 vs EPIC CAFs", "")

# ======================= E2  exact decomposition =======================
print("\n--- E2  exact covariance decomposition (Pearson) ---")
cT = d.DEPTOR_tumor_frac * d.DEPTOR
cM = d.DEPTOR_myeloid_frac * d.DEPTOR
cI = (d.DEPTOR_immune_frac - d.DEPTOR_myeloid_frac) * d.DEPTOR
cO = (1 - d.DEPTOR_tumor_frac - d.DEPTOR_immune_frac) * d.DEPTOR
print(f"    identity check: max|sum(contrib) - DEPTOR| = "
      f"{np.abs((cT+cM+cI+cO) - d.DEPTOR).max():.2e}")
nk = d.NK_of_all
sd_b, sd_n = d.DEPTOR.std(ddof=1), nk.std(ddof=1)
r_obs, p_obs = pearsonr(d.DEPTOR, nk)
parts = {}
for nm, c in [("tumour", cT), ("myeloid", cM),
              ("non-myeloid immune", cI), ("stroma/endothelial", cO)]:
    cv = np.cov(c, nk, ddof=1)[0, 1]
    parts[nm] = cv / (sd_b * sd_n)
    add("E2", f"covariance share: {nm}", len(d), parts[nm], np.nan,
        f"{100*parts[nm]/r_obs:.0f}% of observed r")
add("E2", "SUM of shares", len(d), sum(parts.values()), np.nan, "= observed r")
add("E2", "observed Pearson r", len(d), r_obs, p_obs, "")
print("    NOTE (from E0): contrib_c = (composition-only fraction) x DEPTOR,")
print("    so this decomposition CANNOT detect a myeloid-specific effect.")

# ======================= E3  stratification =======================
print("\n--- E3  stratified by immune content / tumour purity ---")
srows = []


def strat(proxy, label, yvar, xvar="DEPTOR"):
    sub = d[[proxy, xvar, yvar]].dropna().copy()
    sub["grp"] = pd.qcut(sub[proxy], 3, labels=["T1", "T2", "T3"])
    out = []
    for g in ["T1", "T2", "T3"]:
        s = sub[sub.grp == g]
        if len(s) < 20:
            continue
        r, p = spearmanr(s[xvar], s[yvar])
        n = len(s); se = 1 / np.sqrt(n - 3)
        out.append(dict(proxy=label, y=yvar, x=xvar, stratum=g, n=n, rho=r, p=p,
                        ci_lo=np.tanh(np.arctanh(r) - 1.96 * se),
                        ci_hi=np.tanh(np.arctanh(r) + 1.96 * se)))
        print(f"    {label:24s} {yvar:14s} {g}  n={n:3d} rho={r:+.3f} "
              f"({np.tanh(np.arctanh(r)-1.96*se):+.3f},"
              f"{np.tanh(np.arctanh(r)+1.96*se):+.3f}) p={p:.3g}")
    if len(out) >= 2:
        z = np.array([np.arctanh(o["rho"]) for o in out])
        w = np.array([o["n"] - 3 for o in out])
        zm = (z * w).sum() / w.sum()
        Q = float(((z - zm) ** 2 * w).sum())
        pq = 1 - chi2.cdf(Q, len(z) - 1)
        for o in out:
            o["Q"] = Q; o["Q_p"] = pq
        print(f"      -> heterogeneity Q={Q:.2f} (df={len(z)-1}) P={pq:.4g}")
    return out


for proxy, label in [("CD45_expr", "CD45 expression (immune)"),
                     ("EPCAM_expr", "EPCAM expression (tumour)"),
                     ("KRT_expr", "KRT5/14 expression (tumour)"),
                     ("prop_Immune", "custom immune content"),
                     ("prop_Malignant", "custom purity"),
                     ("otherCells", "EPIC otherCells")]:
    for yv in ["NK_of_all", "NK_of_immune"]:
        srows += strat(proxy, label, yv)
sdf = pd.DataFrame(srows)
sdf.to_csv(os.path.join(OUT, "stratified_correlations.csv"), index=False)

# ======================= E4  NK reliability =======================
print("\n--- E4  reliability of the bulk NK estimate ---")
methods = {"custom prop_NK": d.prop_NK, "EPIC NKcells": d.NKcells,
           "quanTIseq NK.cells": d["NK.cells"]}
rel = []
for i, (n1, v1) in enumerate(methods.items()):
    for n2, v2 in list(methods.items())[i + 1:]:
        r, p = spearmanr(v1, v2)
        rel.append(dict(pair=f"{n1} vs {n2}", rho=r, p=p))
        add("E4", f"NK concordance: {n1} vs {n2}", len(d), r, p, "")
print("    attenuation of a true rho=-0.68 by an outcome with reliability R:")
for R in sorted([x["rho"] for x in rel]):
    add("E4", f"attenuation at reliability {R:.3f}", len(d),
        -0.68 * np.sqrt(R), np.nan, f"observed would be {-0.68*np.sqrt(R):+.3f}")
pd.DataFrame(rel).to_csv(os.path.join(OUT, "nk_method_concordance.csv"), index=False)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "bulk_vs_sc_audit.csv"), index=False)
print(f"\nsaved {len(df)} statistics")

# ======================= FIGURE =======================
fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
BLUE, RED, GREY, TEAL, AMB = "#2b6cb0", "#c0392b", "#5a5a5a", "#1f8a70", "#b8860b"

# -- A: stratified forest, immune content, DEPTOR vs NK_of_all --
ax = axes[0, 0]
sub = sdf[(sdf.proxy == "CD45 expression (immune)") & (sdf.y == "NK_of_all") &
          (sdf.x == "DEPTOR")].sort_values("stratum")
ys, labs = [], []
for i, (_, r) in enumerate(sub.iterrows()):
    yy = len(sub) - i; ys.append(yy)
    labs.append(f"{r.stratum}  CD45 {'low' if r.stratum=='T1' else ('mid' if r.stratum=='T2' else 'high')}  (n={r.n})")
    ax.plot([r.ci_lo, r.ci_hi], [yy, yy], color=BLUE, lw=2.2)
    ax.scatter([r.rho], [yy], c=BLUE, s=45, zorder=3)
ax.axvline(0, color="k", lw=1, ls="--")
ax.axvline(-0.68, color=RED, lw=1.5, ls=":")
ax.text(-0.68, 3.62, "scRNA\n−0.68", color=RED, fontsize=7.6, ha="center",
        va="center", linespacing=1.3)
ax.set_yticks(ys); ax.set_yticklabels(labs, fontsize=8)
ax.set_xlabel("Spearman ρ  (DEPTOR vs NK fraction of all cells)", fontsize=9)
ax.set_title("A   Stratified by CD45 (deconvolution-free immune content)",
             fontsize=10.5, fontweight="bold", loc="left")
ax.text(0.985, 0.04, f"heterogeneity across tertiles: P = {sub.Q_p.iloc[0]:.3f}",
        transform=ax.transAxes, fontsize=7.6, color=GREY, ha="right")
ax.tick_params(labelsize=8)
ax.set_xlim(-0.80, 0.46)
ax.set_ylim(0.45, 3.95)

# -- B: NK estimator concordance --
ax = axes[1, 0]
pr = [x["pair"].replace(" vs ", "\nvs\n") for x in rel]
pv = [x["rho"] for x in rel]
ax.bar(range(len(pr)), pv, color=[AMB, AMB, RED], alpha=.88)
ax.axhline(0, color="k", lw=1)
ax.set_xticks(range(len(pr))); ax.set_xticklabels(pr, fontsize=7.4)
ax.set_ylabel("Spearman ρ between NK estimates", fontsize=9)
ax.set_ylim(0, max(pv) * 1.45)
for i, v in enumerate(pv):
    ax.text(i, v + 0.008, f"{v:.3f}", ha="center", fontsize=8)
ax.set_title("B   The bulk NK estimate is barely reproducible",
             fontsize=10.5, fontweight="bold", loc="left")
ax.text(0.5, 0.55, "three deconvolutions of the SAME 515 tumours;\n"
                   "EPIC and quanTIseq agree at ρ = 0.004",
        transform=ax.transAxes, ha="center", fontsize=7.5, color=GREY,
        bbox=dict(fc="#fafafa", ec="#dddddd", lw=.7, pad=4))
ax.tick_params(labelsize=8)

# -- C: identifiability degeneracy --
ax = axes[0, 1]
ax.scatter(et, em, s=9, c=BLUE, alpha=.55, linewidths=0)
lim = [min(et.min(), em.min()), max(et.max(), em.max())]
ax.plot(lim, [lim[0] * (em / et).mean(), lim[1] * (em / et).mean()],
        color=RED, lw=1.6, ls="--")
ax.set_xlabel("tumour-compartment DEPTOR proxy", fontsize=9)
ax.set_ylabel("myeloid-compartment DEPTOR proxy", fontsize=9)
ax.set_title("C   Bulk 'compartment DEPTOR' is composition-only",
             fontsize=10.5, fontweight="bold", loc="left")
ax.text(0.03, 0.92, "r = 1.000 · SD( ratio ) = 0.000\n"
                    "all compartments are the same variable\n"
                    "(constant reference weights)",
        transform=ax.transAxes, fontsize=7.6, color=RED, va="top",
        bbox=dict(fc="white", ec="#dddddd", lw=.7, pad=4))
ax.tick_params(labelsize=8)

# -- D: effect sizes side by side --
ax = axes[1, 1]
imm = sdf[(sdf.proxy == "CD45 expression (immune)") & (sdf.y == "NK_of_all")]
labs = ["scRNA\nmyeloid axis", "scRNA\nmalignant axis", "bulk\nall tumours",
        "bulk\nCD45-high T3", "bulk\nCD45-low T1"]
vals = [-0.680, -0.511,
        float(spearmanr(d.DEPTOR, d.NK_of_all)[0]),
        float(imm[imm.stratum == "T3"].rho.iloc[0]),
        float(imm[imm.stratum == "T1"].rho.iloc[0])]
cols = [RED, RED, BLUE, TEAL, AMB]
ax.bar(range(len(labs)), vals, color=cols, alpha=.88)
ax.axhline(0, color="k", lw=1)
ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, fontsize=7.6)
ax.set_ylabel("Spearman ρ (DEPTOR vs NK)", fontsize=9)
for i, v in enumerate(vals):
    ax.text(i, v + (0.02 if v >= 0 else -0.035), f"{v:+.3f}", ha="center",
            fontsize=8)
ax.set_ylim(-0.85, 0.42)
ax.set_title("D   Single-cell vs bulk effect sizes", fontsize=10.5,
             fontweight="bold", loc="left")
ax.tick_params(labelsize=8)

fig.suptitle("Why the DEPTOR→NK link is strong in sorted single cells but null in "
             "TCGA bulk", fontsize=12.5, fontweight="bold", y=0.975)
fig.text(0.02, 0.012,
         "E0 identifiability · E1 denominator · E2 dilution (exact decomposition) · "
         "E3 stratum · E4 measurement reliability",
         fontsize=8, color=GREY, style="italic")
plt.tight_layout(rect=(0, 0.03, 1, 0.95))
fp = os.path.join(OUT, "Figure_bulk_vs_singlecell.png")
plt.savefig(fp, dpi=300, facecolor="white", bbox_inches="tight")
plt.close()
print("saved", fp)
