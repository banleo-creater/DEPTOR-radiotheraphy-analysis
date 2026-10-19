#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GSE41613 (n=97, HPV-negative OSCC, GPL570) -- THIRD independent cohort.

Goals
 1. deconvolve bulk into 7 compartments with the validated linear-scale solver
 2. split bulk DEPTOR into tumour-attributable vs immune-attributable parts
    (model-driven contributions  +  data-driven regression decomposition)
 3. survival / treatment(uni vs multi modality ~ radiotherapy) interaction
 4. mediation: DEPTOR -> hypoxia -> treatment benefit     (interaction attenuation
    + bootstrap CI, plus a classic a/b/c' mediation inside the RT arm)
"""
import os, sys, gzip, json
import numpy as np, pandas as pd
from scipy.optimize import nnls
from scipy.stats import spearmanr, pearsonr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
import warnings; warnings.filterwarnings("ignore")

BASE = "WorkBuddy/2026-10-17-11-15-46"
sys.path.insert(0, os.path.join(BASE, "scripts"))
from deconv_v2 import deconvolve, REF_MK, score

GEO = "<PATH>"
OUT = os.path.join(BASE, "data", "results_deconv41613"); os.makedirs(OUT, exist_ok=True)
COHORT = "GSE41613"

HYPOXIA = ["HIF1A", "CA9", "VEGFA", "SLC2A1", "LDHA", "PGK1", "ADM", "BNIP3", "PDK1"]
NKSIG = ["NKG7", "GNLY", "KLRD1", "PRF1", "FCGR3A", "NCAM1", "KLRF1", "XCL1", "XCL2", "TBX21", "EOMES"]
TUMOR_MK = ["KRT5", "KRT6A", "KRT14", "KRT17", "KRT13", "SFN", "TP63", "KRT19"]

# ------------------------------------------------------------------ IO
def gpl570_symbols(annot_gz):
    rows = {}
    with gzip.open(annot_gz, "rt", errors="ignore") as f:
        intab = False; sym_col = None
        for line in f:
            if line.startswith("!platform_table_begin"): intab = True; continue
            if line.startswith("!platform_table_end"): break
            if not intab: continue
            p = line.rstrip("\n").split("\t")
            if sym_col is None:
                cols = [c.strip().lower() for c in p]
                for cand in ("gene symbol", "symbol", "gene_symbol"):
                    if cand in cols: sym_col = cols.index(cand); break
                if sym_col is None: sym_col = 2
                continue
            if len(p) <= sym_col: continue
            pid = p[0].strip().strip('"'); sym = p[sym_col].strip().strip('"')
            if pid and sym and sym not in ("", "---"):
                rows[pid] = sym.split("///")[0].strip()
    return rows


def parse_matrix(sm_gz):
    with gzip.open(sm_gz, "rt", errors="ignore") as f:
        lines = f.readlines()
    s = next(i for i, l in enumerate(lines) if l.startswith("!series_matrix_table_begin"))
    e = next(i for i, l in enumerate(lines) if l.startswith("!series_matrix_table_end"))
    hdr = [h.strip().strip('"') for h in lines[s + 1].rstrip("\n").split("\t")]
    ids, data = [], []
    for l in lines[s + 2:e]:
        p = l.rstrip("\n").split("\t")
        ids.append(p[0].strip().strip('"'))
        data.append([float(x) if x.strip() not in ("", "NA", "null", "NaN") else np.nan for x in p[1:]])
    return pd.DataFrame(data, index=ids, columns=hdr[1:])


print(f"===== {COHORT} =====")
p2s = gpl570_symbols(os.path.join(BASE, "data", "plat", "GPL570.annot.gz"))
print("probe->symbol:", len(p2s))
raw = parse_matrix(os.path.join(GEO, COHORT, f"{COHORT}_series_matrix.txt.gz"))
print("probes x samples:", raw.shape)
raw["symbol"] = [p2s.get(i) for i in raw.index]
raw = raw.dropna(subset=["symbol"])
expr = raw.groupby("symbol").max().T            # samples x genes, log2
print("gene-level:", expr.shape)

clin = pd.read_csv(os.path.join(BASE, "data", "gse41613_clinical.csv"), index_col=0)
clin = clin.reindex(expr.index)
clin["os_t"] = pd.to_numeric(clin["fu time"], errors="coerce")
clin["os_e"] = (clin["vital"].astype(str).str.lower().str.startswith("dead")).astype(int)
clin["multi"] = (clin["treatment"].astype(str).str.lower().str.contains("multi")).astype(int)
clin.loc[clin["treatment"].astype(str).str.lower().eq("unknown"), "multi"] = np.nan
clin["stage3"] = (clin["tumor stage"].astype(str).str.contains("III")).astype(int)

# ------------------------------------------------------------------ deconvolution
ref = pd.read_csv(os.path.join(BASE, "data", "gse164690_centroid_full.csv"), index_col=0)
d_ref = ref["DEPTOR"].astype(float)                     # per-cell DEPTOR (log1p CP10K)
CT = ["Malignant", "Stroma", "Endo", "Tcell", "NK", "Bcell", "Myeloid"]

CFG_MAIN = dict(per_class=150, weight="norm")
CFG_SENS = dict(per_class=100, weight="norm")

print("\n-- deconvolution (main cfg) --")
frac, ng = deconvolve(expr, None, ref, verbose=True, **CFG_MAIN)
frac_s, _ = deconvolve(expr, None, ref, verbose=False, **CFG_SENS)
print("  mean:", frac.mean().round(3).to_dict())

# ------------------------------------------------------------------ scores
out = pd.DataFrame(index=expr.index)
out["DEPTOR"] = expr["DEPTOR"]
out["hypoxia"] = score(expr, HYPOXIA)
out["NK_sig"] = score(expr, NKSIG)
out["purity_marker"] = score(expr, TUMOR_MK)
out["immune_sig"] = score(expr, sum([REF_MK[k] for k in ("Tcell", "NK", "Bcell", "Myeloid")], []))
out["stroma_sig"] = score(expr, REF_MK["Stroma"])
out["Tcell_sig"] = score(expr, REF_MK["Tcell"])
out["myeloid_sig"] = score(expr, REF_MK["Myeloid"])
for c in CT:
    out[f"prop_{c}"] = frac[c]
    out[f"propS_{c}"] = frac_s[c]
out["prop_Immune"] = frac[["Tcell", "NK", "Bcell", "Myeloid"]].sum(axis=1)

# ---- model-driven split: contribution of compartment c to bulk DEPTOR
contrib = pd.DataFrame({c: frac[c] * np.expm1(d_ref[c]) for c in CT}, index=frac.index)
tot = contrib.sum(axis=1).replace(0, np.nan)
out["DEPTOR_tumor_frac"] = contrib["Malignant"] / tot
out["DEPTOR_immune_frac"] = contrib[["Tcell", "NK", "Bcell", "Myeloid"]].sum(axis=1) / tot
out["DEPTOR_myeloid_frac"] = contrib["Myeloid"] / tot
out["DEPTOR_stroma_frac"] = contrib["Stroma"] / tot
out["DEPTOR_tumor_contrib"] = contrib["Malignant"]
out["DEPTOR_immune_contrib"] = contrib[["Tcell", "NK", "Bcell", "Myeloid"]].sum(axis=1)
# sensitivity cfg shares
contrib_s = pd.DataFrame({c: frac_s[c] * np.expm1(d_ref[c]) for c in CT}, index=frac_s.index)
tots = contrib_s.sum(axis=1).replace(0, np.nan)
out["DEPTOR_tumor_frac_S"] = contrib_s["Malignant"] / tots
out["DEPTOR_immune_frac_S"] = contrib_s[["Tcell", "NK", "Bcell", "Myeloid"]].sum(axis=1) / tots

for c in ["os_t", "os_e", "multi", "stage3", "Sex", "age"]:
    out[c] = clin[c]
out["treatment"] = clin["treatment"]
out["vital"] = clin["vital"]
out.to_csv(os.path.join(OUT, f"{COHORT}_deconv_sample.csv"))
print("\nDEPTOR split (mean):", out[["DEPTOR_tumor_frac", "DEPTOR_immune_frac",
                                    "DEPTOR_myeloid_frac", "DEPTOR_stroma_frac"]].mean().round(4).to_dict())

# ---- data-driven check: how does bulk DEPTOR split along the tumour / immune axis?
# (a 7-component regression is not identifiable because the fractions are nearly
#  collinear and sum to 1; use a 2-block model instead)
blk = out[["DEPTOR", "purity_marker", "immune_sig", "stroma_sig"]].dropna().copy()
for c in ["DEPTOR", "purity_marker", "immune_sig", "stroma_sig"]:
    blk[c + "_z"] = (blk[c] - blk[c].mean()) / blk[c].std()
import numpy.linalg as la
Xb = np.column_stack([np.ones(len(blk)), blk["purity_marker_z"], blk["immune_sig_z"]])
bb, *_ = la.lstsq(Xb, blk["DEPTOR_z"].values, rcond=None)
pred = Xb @ bb
r2 = 1 - ((blk["DEPTOR_z"].values - pred) ** 2).sum() / ((blk["DEPTOR_z"].values - blk["DEPTOR_z"].mean()) ** 2).sum()
print("\n2-block variance decomposition  DEPTOR_z ~ purity_z + immune_z")
print(f"  beta_purity={bb[1]:+.3f}   beta_immune={bb[2]:+.3f}   R2={r2:.3f}   n={len(blk)}")
pd.DataFrame([dict(term=["intercept", "purity_marker_z", "immune_sig_z"], beta=bb,
                   R2=r2, n=len(blk))]).to_csv(
    os.path.join(OUT, f"{COHORT}_DEPTOR_variance_decomposition.csv"), index=False)

# ------------------------------------------------------------------ correlations
print("\n== correlations with bulk DEPTOR ==")
crows = []
for v in ["hypoxia", "NK_sig", "purity_marker", "immune_sig", "stroma_sig", "myeloid_sig",
          "prop_Malignant", "prop_Immune", "prop_Myeloid",
          "DEPTOR_tumor_frac", "DEPTOR_immune_frac", "DEPTOR_myeloid_frac"]:
    r, p = spearmanr(out["DEPTOR"], out[v], nan_policy="omit")
    crows.append(dict(variable=v, rho=round(r, 3), p=round(p, 4), n=int(out[[v]].dropna().shape[0])))
    print(f"  DEPTOR ~ {v:22s} rho={r:+.3f} p={p:.2g}")
pd.DataFrame(crows).to_csv(os.path.join(OUT, f"{COHORT}_correlations.csv"), index=False)

# ------------------------------------------------------------------ survival
def cox(d, covs, end_t="os_t", end_e="os_e"):
    d = d.dropna(subset=[end_t, end_e] + covs).copy()
    if len(d) < 25 or d[end_e].sum() < 5: return None
    cph = CoxPHFitter()
    cph.fit(d[[end_t, end_e] + covs], duration_col=end_t, event_col=end_e)
    return cph, d


def zscore(d, cols):
    for c in cols:
        d[c + "_z"] = (d[c] - d[c].mean()) / d[c].std()
    return d


print("\n== Cox (per SD) ==")
d = zscore(out.copy(), ["DEPTOR", "DEPTOR_tumor_frac", "DEPTOR_immune_frac",
                        "DEPTOR_myeloid_frac", "hypoxia", "NK_sig", "purity_marker"])
rows = []
for v in ["DEPTOR", "DEPTOR_tumor_frac", "DEPTOR_immune_frac", "DEPTOR_myeloid_frac",
          "hypoxia", "NK_sig", "purity_marker"]:
    r = cox(d, [v + "_z"])
    if r is None: continue
    cph, dd = r; s = cph.summary.loc[v + "_z"]
    rows.append(dict(var=v, HR=round(s["exp(coef)"], 3), lo=round(s["exp(coef) lower 95%"], 3),
                     hi=round(s["exp(coef) upper 95%"], 3), p=round(s["p"], 4), n=len(dd),
                     events=int(dd.os_e.sum())))
    print(f"  {v:22s} HR/SD={s['exp(coef)']:.2f} ({s['exp(coef) lower 95%']:.2f}-"
          f"{s['exp(coef) upper 95%']:.2f}) p={s['p']:.3g} n={len(dd)} ev={int(dd.os_e.sum())}")
pd.DataFrame(rows).to_csv(os.path.join(OUT, f"{COHORT}_cox.csv"), index=False)

print("\n== Cox by treatment arm ==")
srows = []
for arm, lbl in [(0, "uni-modality (no RT)"), (1, "multi-modality (+RT)")]:
    sub = d[d.multi == arm]
    for v in ["DEPTOR", "DEPTOR_tumor_frac", "DEPTOR_immune_frac"]:
        r = cox(sub, [v + "_z"])
        if r is None: continue
        cph, dd = r; s = cph.summary.loc[v + "_z"]
        srows.append(dict(arm=lbl, var=v, HR=round(s["exp(coef)"], 3),
                          lo=round(s["exp(coef) lower 95%"], 3), hi=round(s["exp(coef) upper 95%"], 3),
                          p=round(s["p"], 4), n=len(dd), events=int(dd.os_e.sum())))
        print(f"  {lbl:22s} {v:20s} HR={s['exp(coef)']:.2f} "
              f"({s['exp(coef) lower 95%']:.2f}-{s['exp(coef) upper 95%']:.2f}) "
              f"p={s['p']:.3g} n={len(dd)} ev={int(dd.os_e.sum())}")
pd.DataFrame(srows).to_csv(os.path.join(OUT, f"{COHORT}_cox_by_arm.csv"), index=False)

print("\n== interaction DEPTOR x treatment (unadjusted / adjusted for stage) ==")
irows = []
for v in ["DEPTOR", "DEPTOR_tumor_frac", "DEPTOR_immune_frac"]:
    d2 = d.dropna(subset=["os_t", "os_e", "multi", v + "_z"]).copy()
    d2["inter"] = d2[v + "_z"] * d2["multi"]
    for covs, tag in [([v + "_z", "multi", "inter"], "unadj"),
                      ([v + "_z", "multi", "inter", "stage3"], "adj_stage")]:
        cph = CoxPHFitter()
        cph.fit(d2[["os_t", "os_e"] + covs], duration_col="os_t", event_col="os_e")
        s = cph.summary.loc["inter"]
        irows.append(dict(var=v, model=tag, inter_HR=round(s["exp(coef)"], 3),
                          lo=round(s["exp(coef) lower 95%"], 3), hi=round(s["exp(coef) upper 95%"], 3),
                          p=round(s["p"], 4), n=len(d2), events=int(d2.os_e.sum()),
                          logHR=round(s["coef"], 4), se=round(s["se(coef)"], 4)))
        print(f"  {v:20s} {tag:10s} inter HR={s['exp(coef)']:.2f} "
              f"({s['exp(coef) lower 95%']:.2f}-{s['exp(coef) upper 95%']:.2f}) p={s['p']:.3g}")
pd.DataFrame(irows).to_csv(os.path.join(OUT, f"{COHORT}_interaction.csv"), index=False)

# ------------------------------------------------------------------ mediation
print("\n== mediation: DEPTOR -> hypoxia -> treatment benefit ==")
med = []
for v in ["DEPTOR", "DEPTOR_tumor_frac"]:
    d2 = d.dropna(subset=["os_t", "os_e", "multi", v + "_z", "hypoxia_z"]).copy()
    if len(d2) < 50: continue
    d2["inter"] = d2[v + "_z"] * d2["multi"]
    # a path: X -> M
    a = np.polyfit(d2[v + "_z"], d2["hypoxia_z"], 1)[0]
    ra, pa = spearmanr(d2[v + "_z"], d2["hypoxia_z"])
    # total interaction
    c1 = CoxPHFitter(); c1.fit(d2[["os_t", "os_e", v + "_z", "multi", "inter"]],
                               duration_col="os_t", event_col="os_e")
    th1 = c1.summary.loc["inter", "coef"]
    # adjusted for mediator
    c2 = CoxPHFitter(); c2.fit(d2[["os_t", "os_e", v + "_z", "multi", "inter", "hypoxia_z"]],
                               duration_col="os_t", event_col="os_e")
    th2 = c2.summary.loc["inter", "coef"]
    bh = c2.summary.loc["hypoxia_z"]
    # bootstrap the mediated proportion
    rng = np.random.default_rng(7)
    props = []
    for _ in range(300):
        b = d2.sample(len(d2), replace=True, random_state=int(rng.integers(1e6)))
        try:
            m1 = CoxPHFitter().fit(b[["os_t", "os_e", v + "_z", "multi", "inter"]],
                                   duration_col="os_t", event_col="os_e")
            m2 = CoxPHFitter().fit(b[["os_t", "os_e", v + "_z", "multi", "inter", "hypoxia_z"]],
                                   duration_col="os_t", event_col="os_e")
            t1 = m1.summary.loc["inter", "coef"]; t2 = m2.summary.loc["inter", "coef"]
            if abs(t1) > 1e-6: props.append((t1 - t2) / t1)
        except Exception:
            pass
    lo, hi = (np.percentile(props, [2.5, 97.5]) if len(props) > 30 else (np.nan, np.nan))
    med.append(dict(X=v, n=len(d2), a_path=round(a, 3), a_rho=round(ra, 3), a_p=round(pa, 4),
                    inter_logHR_total=round(th1, 4), inter_logHR_adj=round(th2, 4),
                    b_hypoxia_logHR=round(bh["coef"], 4), b_p=round(bh["p"], 4),
                    mediated_prop=round((th1 - th2) / th1, 3) if abs(th1) > 1e-6 else np.nan,
                    boot_lo=round(lo, 3) if lo == lo else np.nan,
                    boot_hi=round(hi, 3) if hi == hi else np.nan,
                    n_boot=len(props)))
    print(f"  X={v:20s} a={a:+.3f} (rho={ra:+.3f},p={pa:.2g}) | inter logHR total={th1:+.3f} "
          f"-> adj={th2:+.3f} | b(hypoxia)={bh['coef']:+.3f}(p={bh['p']:.2g}) | "
          f"mediated={((th1-th2)/th1 if abs(th1)>1e-6 else float('nan')):.1%} "
          f"[{lo:.2f},{hi:.2f}]")
pd.DataFrame(med).to_csv(os.path.join(OUT, f"{COHORT}_mediation_interaction.csv"), index=False)

# classic mediation inside the RT arm
print("\n  -- classic a/b/c' inside RT(+) arm --")
m2rows = []
rt = d[d.multi == 1]
for v in ["DEPTOR", "DEPTOR_tumor_frac"]:
    dd = rt.dropna(subset=["os_t", "os_e", v + "_z", "hypoxia_z"]).copy()
    if len(dd) < 40: continue
    a = np.polyfit(dd[v + "_z"], dd["hypoxia_z"], 1)[0]
    c1 = CoxPHFitter(); c1.fit(dd[["os_t", "os_e", v + "_z"]], duration_col="os_t", event_col="os_e")
    tot = c1.summary.loc[v + "_z", "coef"]
    c2 = CoxPHFitter(); c2.fit(dd[["os_t", "os_e", v + "_z", "hypoxia_z"]], duration_col="os_t", event_col="os_e")
    dir_ = c2.summary.loc[v + "_z", "coef"]
    m2rows.append(dict(X=v, n=len(dd), events=int(dd.os_e.sum()), a_path=round(a, 3),
                       total_logHR=round(tot, 4), direct_logHR=round(dir_, 4),
                       mediated_prop=round((tot - dir_) / tot, 3) if abs(tot) > 1e-6 else np.nan))
    print(f"  {v:20s} n={len(dd)} a={a:+.3f} total={tot:+.3f} direct={dir_:+.3f} "
          f"mediated={((tot-dir_)/tot if abs(tot)>1e-6 else float('nan')):.1%}")
pd.DataFrame(m2rows).to_csv(os.path.join(OUT, f"{COHORT}_mediation_RTarm.csv"), index=False)

print("\nSaved ->", OUT)
