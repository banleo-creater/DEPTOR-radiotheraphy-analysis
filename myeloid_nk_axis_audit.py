#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Full statistical audit of the DEPTOR -> myeloid -> NK axis.

Collects, and where missing RECOMPUTES, every statistic that supports (or fails
to support) the claim:

    "DEPTOR-high myeloid niche  ==  NK-poor microenvironment"

Evidence layers
  L1 scRNA patient-level  : GSE164690 (myeloid/M2/cDC DEPTOR vs NK fraction)
  L2 bulk pseudobulk      : GSE139324, GSE198315 (external replication)
  L3 TCGA bulk            : DEPTOR vs deconvolved NK fraction  (RECOMPUTED HERE,
                            incl. purity-adjusted and HPV-stratified)
  L4 spatial single-cell  : GSE300147 Xenium (read from results_xenium_mnk)
  L5 spatial spot         : GSE181300 Visium (tumour-core NK desert)

Outputs a single tidy CSV + a Markdown table for the manuscript.
"""
import os, warnings
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr
warnings.filterwarnings("ignore")

BASE = "WorkBuddy/2026-10-17-11-15-46"
OUT = os.path.join(BASE, "data", "results_myeloid_nk_axis")
os.makedirs(OUT, exist_ok=True)

rows = []


def add(layer, cohort, test, n, est, p, extra=""):
    rows.append(dict(layer=layer, cohort=cohort, test=test, n=n,
                     estimate=est, p=p, note=extra))


# ---------------------------------------------------------------- L3: TCGA
print("=== L3: TCGA bulk DEPTOR vs deconvolved NK fraction ===")
d = pd.read_csv(os.path.join(BASE, "data", "results_deconv", "deconv_per_sample.csv"))
d = d.dropna(subset=["DEPTOR", "prop_NK"]).copy()
print(f"  n with DEPTOR + prop_NK = {len(d)}")

for lab, col in [("DEPTOR (bulk)", "DEPTOR"),
                 ("DEPTOR_myeloid_frac", "DEPTOR_myeloid_frac"),
                 ("DEPTOR_immune_frac", "DEPTOR_immune_frac")]:
    if col not in d.columns:
        print(f"  (skip {col}: absent)"); continue
    s = d.dropna(subset=[col])
    r, p = spearmanr(s[col], s["prop_NK"])
    rp, pp = pearsonr(s[col], s["prop_NK"])
    add("L3 TCGA bulk", "TCGA-HNSC", f"{lab} vs NK fraction (Spearman)",
        len(s), r, p, f"Pearson r={rp:+.3f}, p={pp:.3g}")
    print(f"  {lab:22s} vs prop_NK : rho={r:+.3f} p={p:.3g}  (n={len(s)}, Pearson r={rp:+.3f})")

# purity-adjusted: residualise both on malignant fraction
if "prop_Malignant" in d.columns:
    s = d.dropna(subset=["DEPTOR", "prop_NK", "prop_Malignant"])
    X = np.column_stack([np.ones(len(s)), s["prop_Malignant"].values])
    def resid(y):
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        return y - X @ b
    rd, rnk = resid(s["DEPTOR"].values), resid(s["prop_NK"].values)
    r, p = spearmanr(rd, rnk)
    add("L3 TCGA bulk", "TCGA-HNSC", "DEPTOR vs NK fraction, purity-adjusted (Spearman)",
        len(s), r, p, "both residualised on prop_Malignant")
    print(f"  purity-adjusted            : rho={r:+.3f} p={p:.3g} (n={len(s)})")
    # partial Pearson
    rp, pp = pearsonr(rd, rnk)
    add("L3 TCGA bulk", "TCGA-HNSC", "DEPTOR vs NK fraction, purity-adjusted (Pearson)",
        len(s), rp, pp, "both residualised on prop_Malignant")

# HPV-stratified TCGA
if "HPV" in d.columns:
    for h in sorted(d["HPV"].dropna().unique()):
        s = d[d["HPV"] == h].dropna(subset=["DEPTOR", "prop_NK"])
        if len(s) < 20: continue
        r, p = spearmanr(s["DEPTOR"], s["prop_NK"])
        add("L3 TCGA bulk", "TCGA-HNSC", f"DEPTOR vs NK fraction, HPV={h}",
            len(s), r, p, "HPV-stratified")
        print(f"  HPV={h:5s}                 : rho={r:+.3f} p={p:.3g} (n={len(s)})")

# NK fraction by HPV (already reported, re-checked here)
if "HPV" in d.columns:
    g = d.dropna(subset=["HPV", "prop_NK"]).groupby("HPV")["prop_NK"]
    m = g.mean(); s = g.std(); n = g.count()
    print("  NK fraction by HPV:", {k: round(v, 4) for k, v in m.items()}, dict(n))
    for h in m.index:
        add("L3 TCGA bulk", "TCGA-HNSC", f"NK fraction mean, HPV={h}",
            int(n[h]), float(m[h]), np.nan, f"SD={s[h]:.4f}")

# ------------------------------------------------------- L1/L2: from files
print("\n=== L1/L2: scRNA + pseudobulk (from saved CSVs) ===")
p1 = os.path.join(BASE, "data", "results_myeloid", "myeloid_subtype_correlations.csv")
if os.path.exists(p1):
    c = pd.read_csv(p1)
    for _, r in c.iterrows():
        if pd.isna(r.get("Spearman_rho")): continue
        add("L1 scRNA", "GSE164690", f"{r['comparison']} (Spearman)",
            int(r["n"]), float(r["Spearman_rho"]), float(r["Spearman_p"]),
            f"Pearson r={r['Pearson_r']}")
        print(f"  GSE164690 {r['comparison']:44s} rho={r['Spearman_rho']:+.3f} p={r['Spearman_p']:.4g}")

p2 = os.path.join(BASE, "data", "results_multicohort", "multicohort_correlations.csv")
if os.path.exists(p2):
    c = pd.read_csv(p2)
    for _, r in c.iterrows():
        add("L2 pseudobulk", r["cohort"], "myeloid DEPTOR vs NK fraction (Spearman)",
            int(r["n"]), float(r["Spearman_rho"]), float(r["Spearman_p"]),
            f"Pearson r={r['Pearson_r']}")
        print(f"  {r['cohort']:12s} myeloid DEPTOR vs NK                      "
              f"rho={r['Spearman_rho']:+.3f} p={r['Spearman_p']:.4g} (n={int(r['n'])})")

# ------------------------------------------------------- L4: Xenium (new)
print("\n=== L4: Xenium single-cell spatial ===")
xn = os.path.join(BASE, "data", "results_xenium_mnk")
for fn, key in [("xenium_mnk_enrich100.csv", "enr"),
                ("xenium_mnk_paired_contrasts.csv", "contrast"),
                ("xenium_mnk_nontumor.csv", "nt"),
                ("xenium_mnk_nontumor_contrasts.csv", "ntc"),
                ("xenium_mnk_nnz_aggregate.csv", "nnz"),
                ("xenium_mnk_hpv.csv", "hpv")]:
    fp = os.path.join(xn, fn)
    if not os.path.exists(fp):
        print(f"  (missing {fn} — run xenium_myeloid_nk_colocalization.py first)"); continue
    c = pd.read_csv(fp)
    for _, r in c.iterrows():
        if key == "nnz":
            add("L4 Xenium spatial", "GSE300147", f"NN z {r['pair']}",
                int(r["n_sections"]), float(r["mean_z"]), float(r["p"]),
                f"z>0 in {r['n_excluded']}/{r['n_sections']}")
            print(f"  NN z {r['pair']:14s} mean z={r['mean_z']:+.2f} p={r['p']:.3g} "
                  f"(n={int(r['n_sections'])})")
        elif key == "contrast":
            add("L4 Xenium spatial", "GSE300147", f"paired {r['contrast']}",
                int(r["n"]), float(r["mean_diff"]), float(r["p"]), "paired t-test")
            print(f"  {r['contrast']:32s} diff={r['mean_diff']:+.2f} p={r['p']:.3g}")
        elif key == "enr":
            add("L4 Xenium spatial", "GSE300147", f"enrichment@100um {r['pair']}",
                int(r["n"]), float(r["mean_enr"]), float(r["p_vs_1"]),
                f"95%CI {r['ci_lo']:.2f}-{r['ci_hi']:.2f}")
            print(f"  enr@100um {r['pair']:14s} = {r['mean_enr']:.3f} p={r['p_vs_1']:.3g}")
        elif key == "nt":
            lab = f"{r['pair']} [{r['kind']}] non-tumour"
            add("L4 Xenium spatial", "GSE300147", lab,
                int(r["n"]), float(r["est"]), float(r["p"]),
                "tumour cells excluded")
            print(f"  non-tumour {lab:38s} = {r['est']:.3f} p={r['p']:.3g}")
        elif key == "ntc":
            add("L4 Xenium spatial", "GSE300147", f"paired {r['contrast']}",
                int(r["n"]), float(r["mean_diff"]), float(r["p"]),
                "paired t-test, non-tumour")
            print(f"  {r['contrast']:44s} diff={r['mean_diff']:+.2f} p={r['p']:.3g}")
        elif key == "hpv":
            add("L4 Xenium spatial", "GSE300147", f"{r['metric']} HPV+ vs HPV-",
                int(r["n_pos"] + r["n_neg"]), float(r["hpv_pos"] - r["hpv_neg"]),
                float(r["MWU_p"]), f"HPV+ {r['hpv_pos']:.3f} vs HPV- {r['hpv_neg']:.3f}")
            print(f"  {r['metric']:14s} HPV+ {r['hpv_pos']:+.3f} vs HPV- {r['hpv_neg']:+.3f} "
                  f"p={r['MWU_p']:.3g}")

# raw-distance HPV contrast (the metric used in the original Figure 6D)
_p = os.path.join(xn, "xenium_mnk_persection.csv")
if os.path.exists(_p):
    from scipy.stats import mannwhitneyu
    d2 = pd.read_csv(_p)
    for c, lab in [("nnd_Tumor_NK", "tumour->nearest NK distance (um)"),
                   ("nnd_NK_M2", "NK->nearest M2 distance (um)"),
                   ("frac_NK", "NK fraction")]:
        if c not in d2.columns:
            continue
        a = d2.loc[d2.hpv == "HPV+", c].dropna()
        b = d2.loc[d2.hpv == "HPV-", c].dropna()
        if len(a) < 3 or len(b) < 3:
            continue
        _, pv = mannwhitneyu(a, b)
        add("L4 Xenium spatial", "GSE300147", f"{lab}, HPV+ vs HPV-",
            len(a) + len(b), float(a.mean() - b.mean()), float(pv),
            f"HPV+ {a.mean():.2f} vs HPV- {b.mean():.2f}")
        print(f"  {lab:36s} HPV+ {a.mean():7.2f} vs HPV- {b.mean():7.2f}  p={pv:.4g}")

    # paired ENRICHMENT contrast — the metric plotted in panel D
    from scipy.stats import ttest_rel
    cM2, cM1 = "nt_enr_NK_M2_100", "nt_enr_NK_M1_100"
    if cM2 in d2.columns and cM1 in d2.columns:
        mm = d2[[cM2, cM1]].dropna()
        if len(mm) >= 3:
            _, pv = ttest_rel(mm[cM2], mm[cM1])
            add("L4 Xenium spatial", "GSE300147",
                "paired enrichment NK->M2 vs NK->M1 (non-tumour)",
                len(mm), float((mm[cM2] - mm[cM1]).mean()), float(pv),
                f"M2 {mm[cM2].mean():.3f} vs M1 {mm[cM1].mean():.3f}; "
                f"M2<1 in {int((mm[cM2] < 1).sum())}/{len(mm)} sections")
            print(f"  paired enrich NK->M2 vs M1 (non-tumour): "
                  f"diff={(mm[cM2]-mm[cM1]).mean():+.3f} p={pv:.4g}  "
                  f"(M2<1 in {int((mm[cM2]<1).sum())}/{len(mm)})")

# ------------------------------------------------------- L5: Visium
print("\n=== L5: Visium (GSE181300) ===")
add("L5 Visium", "GSE181300", "NK fraction: invasive front vs tumour core",
    12038, 0.034 / 0.017, np.nan, "core NK = 0.017, front NK = 0.034 (ratio ~2x)")
add("L5 Visium", "GSE181300", "DEPTOR+ vs DEPTOR- tumour spot: neighbourhood Mac fraction",
    7765, 0.0436 / 0.0371, 0.0089, "MWU one-sided; DEPTOR+ n=46 (exploratory)")
add("L5 Visium", "GSE181300", "DEPTOR+ vs DEPTOR- tumour spot: neighbourhood NK fraction",
    7765, 0.0227 / 0.0218, 0.953, "NOT replicated (DEPTOR+ n=46)")
print("  (values transcribed from results_spatial/RESULTS_分细胞类型与空间验证.md)")

# ------------------------------------------------------- assemble
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "myeloid_NK_axis_full_audit.csv"), index=False)
print(f"\nsaved {len(df)} statistics -> {OUT}")

# ---- random-effects style summary of the myeloid-DEPTOR -> NK correlation ----
print("\n" + "=" * 70)
print("META-VIEW: myeloid/DEPTOR -> NK correlation across cohorts")
print("=" * 70)
core = df[df["test"].str.contains("myeloid DEPTOR vs NK|myeloid DEPTOR|M2-TAM|cDC|total myeloid",
                                  case=False, regex=True)]
core = core[core["layer"].isin(["L1 scRNA", "L2 pseudobulk"])]
if len(core):
    print(core[["cohort", "test", "n", "estimate", "p"]].to_string(index=False))
    # Fisher-z fixed-effect meta
    z = np.arctanh(np.clip(core["estimate"].values, -0.999, 0.999))
    w = core["n"].values - 3
    zm = float((z * w).sum() / w.sum())
    se = float(1 / np.sqrt(w.sum()))
    print(f"\n  fixed-effect Fisher-z pooled rho = {np.tanh(zm):+.3f} "
          f"(95%CI {np.tanh(zm-1.96*se):+.3f} to {np.tanh(zm+1.96*se):+.3f})")
    from scipy.stats import chi2
    q = float(((z - zm) ** 2 * w).sum())
    het_p = 1 - chi2.cdf(q, len(z) - 1)
    print(f"  heterogeneity Q={q:.2f} (df={len(z)-1}) p={het_p:.3g}")
    pd.DataFrame([dict(pooled_rho=np.tanh(zm),
                       ci_lo=np.tanh(zm - 1.96 * se), ci_hi=np.tanh(zm + 1.96 * se),
                       Q=q, het_p=het_p, k=len(z))]).to_csv(
        os.path.join(OUT, "myeloid_NK_meta.csv"), index=False)
print("\nDONE")
