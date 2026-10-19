#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Audit of the NKG2D-ligand hypothesis analysis (2026-10-19).

Three defects are repaired relative to scripts/nkg2d_nk_survival.py:

1. The original built both programmes as the mean of per-gene z-scores computed
   on RAW RSEM.  Raw RSEM is heavily right-skewed and spans two orders of
   magnitude between genes (KLRF1 mean 5.2 vs FCGR3A 1100), so a mean of raw
   z-scores is effectively a weighted sum dominated by the highest-variance
   gene.  Both programmes are rebuilt on log2(RSEM+1) z-scores, and the raw
   version is retained for side-by-side comparison.
2. The original used the whole-cohort median for the KM quadrants but the
   within-subset median for the Cox models.  The cut-points are now fixed at
   the whole-cohort median everywhere, as in the DEPTOR analysis.
3. The original did not record events or the analytic set, so the numbers could
   not be audited.  Everything is written to tables now.

Outputs (data/results_nkg2d/):
  nkg2d_scores.csv                per-sample programmes, both encodings
  nkg2d_sensitivity.csv           correlation with DEPTOR under both encodings
  nkg2d_quadrant.csv              quadrant sizes and log-rank tests
  nkg2d_cox.csv                   interaction and main-effect Cox models
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
import warnings
warnings.filterwarnings("ignore")

BASE = "WorkBuddy/2026-10-17-11-15-46"
OUT = os.path.join(BASE, "data", "results_nkg2d")

LIG = ["MICA", "MICB", "ULBP1", "ULBP2", "ULBP3", "RAET1E", "RAET1G", "RAET1L"]
NKC = ["NKG7", "GNLY", "KLRD1", "PRF1", "FCGR3A", "NCAM1", "KLRF1",
       "XCL1", "XCL2", "TBX21", "EOMES"]

clin = pd.read_csv(os.path.join(BASE, "data", "tcga_hnsc_pca_deptor.csv"))
nkg = pd.read_csv(os.path.join(BASE, "data", "tcga_hnsc_nkg2d_nk.csv"))
d = clin.merge(nkg, on="sampleId", how="inner")

d["RT"] = d.RADIATION_THERAPY.astype(str).str.strip()
d["DEPTOR"] = pd.to_numeric(d.DEPTOR_RSEM, errors="coerce")
for ep in ["OS", "PFS", "DSS"]:
    d[f"{ep}_MONTHS"] = pd.to_numeric(d[f"{ep}_MONTHS"], errors="coerce")
    s = d[f"{ep}_STATUS"].astype(str).str.upper().str.strip()
    d[f"{ep}_event"] = s.str.startswith("1").astype(int)


def programme(genes, log):
    """Mean of per-gene z-scores, optionally on log2(RSEM+1)."""
    cols = []
    for g in genes:
        v = pd.to_numeric(d[g], errors="coerce")
        if log:
            v = np.log2(v + 1.0)
        d[g + ("_lz" if log else "_rz")] = (v - v.mean()) / v.std(ddof=1)
        cols.append(g + ("_lz" if log else "_rz"))
    return d[cols].mean(axis=1)


for log, tag in [(True, "log"), (False, "raw")]:
    d[f"LIG_{tag}"] = programme(LIG, log)
    d[f"NK_{tag}"] = programme(NKC, log)

d[["sampleId", "RT", "DEPTOR", "LIG_log", "NK_log", "LIG_raw", "NK_raw",
   "OS_MONTHS", "OS_event", "PFS_MONTHS", "PFS_event",
   "DSS_MONTHS", "DSS_event"]].to_csv(
    os.path.join(OUT, "nkg2d_scores.csv"), index=False)

# ------------------------------------------------------------ 1. sensitivity
rows = []
for tag in ["log", "raw"]:
    for score in ["LIG", "NK"]:
        for scope, sub in [("All", d), ("RT=Yes", d[d.RT == "Yes"])]:
            s = sub[["DEPTOR", f"{score}_{tag}"]].dropna()
            r, p = spearmanr(s.iloc[:, 0], s.iloc[:, 1])
            rows.append(dict(encoding=tag, score=score, scope=scope,
                             n=len(s), rho=r, p=p))
sens = pd.DataFrame(rows)
sens.to_csv(os.path.join(OUT, "nkg2d_sensitivity.csv"), index=False)
print("=== DEPTOR vs programmes ===")
print(sens.round(4).to_string(index=False))

# correlation between the two NK encodings, to show how much they differ
s = d[["NK_log", "NK_raw"]].dropna()
print("\nNK_log vs NK_raw: rho = %+.3f (n = %d)"
      % (spearmanr(s.iloc[:, 0], s.iloc[:, 1])[0], len(s)))

# ------------------------------------------------------- 2. quadrant analysis
# cut-points fixed at the whole-cohort median, both encodings
rt = d[d.RT == "Yes"].copy()
qrows = []
for tag in ["log", "raw"]:
    lig_cut = d[f"LIG_{tag}"].median()
    nk_cut = d[f"NK_{tag}"].median()
    rt[f"Q_{tag}"] = np.select(
        [rt[f"LIG_{tag}"] > lig_cut, rt[f"LIG_{tag}"] <= lig_cut],
        ["ligand-high", "ligand-low"], default="NA")
    rt[f"K_{tag}"] = np.select(
        [rt[f"NK_{tag}"] > nk_cut, rt[f"NK_{tag}"] <= nk_cut],
        ["NK-high", "NK-low"], default="NA")
    rt[f"quad_{tag}"] = rt[f"Q_{tag}"] + " / " + rt[f"K_{tag}"]

    for ep in ["OS", "PFS", "DSS"]:
        s = rt.dropna(subset=[f"{ep}_MONTHS", f"quad_{tag}"]).copy()
        tgt = "ligand-high / NK-low"
        a = s[s[f"quad_{tag}"] == tgt]
        b = s[s[f"quad_{tag}"] != tgt]
        lr = logrank_test(a[f"{ep}_MONTHS"], b[f"{ep}_MONTHS"],
                          a[f"{ep}_event"], b[f"{ep}_event"])
        qrows.append(dict(encoding=tag, endpoint=ep, comparison=f"{tgt} vs rest",
                          n_target=len(a), events_target=int(a[f'{ep}_event'].sum()),
                          n_rest=len(b), events_rest=int(b[f'{ep}_event'].sum()),
                          p=lr.p_value))
        for q, g in s.groupby(f"quad_{tag}"):
            qrows.append(dict(encoding=tag, endpoint=ep, comparison=f"size: {q}",
                              n_target=len(g), events_target=int(g[f'{ep}_event'].sum()),
                              n_rest=np.nan, events_rest=np.nan, p=np.nan))
quad = pd.DataFrame(qrows)
quad.to_csv(os.path.join(OUT, "nkg2d_quadrant.csv"), index=False)
print("\n=== quadrant, log encoding, DSS ===")
print(quad[(quad.encoding == "log") & (quad.endpoint == "DSS") &
           (~quad.comparison.str.startswith("size"))].round(4).to_string(index=False))
print("\n=== quadrant sizes (log encoding, DSS) ===")
print(quad[(quad.encoding == "log") & (quad.endpoint == "DSS") &
           (quad.comparison.str.startswith("size"))][["comparison", "n_target", "events_target"]]
      .to_string(index=False))

# ------------------------------------------------------------- 3. Cox models
crows = []
for tag in ["log", "raw"]:
    lig_cut = d[f"LIG_{tag}"].median()
    nk_cut = d[f"NK_{tag}"].median()
    for ep in ["OS", "PFS", "DSS"]:
        s = rt.dropna(subset=[f"{ep}_MONTHS", f"LIG_{tag}", f"NK_{tag}"]).copy()
        s["T"] = s[f"{ep}_MONTHS"]
        s["E"] = s[f"{ep}_event"]
        s["ligand_hi"] = (s[f"LIG_{tag}"] > lig_cut).astype(float)
        s["NK_low"] = (s[f"NK_{tag}"] <= nk_cut).astype(float)
        s["IX"] = s.ligand_hi * s.NK_low
        for term in ["ligand_hi", "NK_low"]:
            c = CoxPHFitter().fit(s[["T", "E", term]], "T", "E")
            r = c.summary.loc[term]
            crows.append(dict(encoding=tag, endpoint=ep, model="univariate",
                              term=term, n=len(s), events=int(s.E.sum()),
                              HR=np.exp(r["coef"]),
                              lo=np.exp(r["coef lower 95%"]),
                              hi=np.exp(r["coef upper 95%"]), p=r["p"]))
        c = CoxPHFitter().fit(s[["T", "E", "ligand_hi", "NK_low", "IX"]], "T", "E")
        for term in ["ligand_hi", "NK_low", "IX"]:
            r = c.summary.loc[term]
            crows.append(dict(encoding=tag, endpoint=ep, model="interaction",
                              term=term, n=len(s), events=int(s.E.sum()),
                              HR=np.exp(r["coef"]),
                              lo=np.exp(r["coef lower 95%"]),
                              hi=np.exp(r["coef upper 95%"]), p=r["p"]))
cox = pd.DataFrame(crows)
cox.to_csv(os.path.join(OUT, "nkg2d_cox.csv"), index=False)
print("\n=== Cox, log encoding ===")
print(cox[cox.encoding == "log"].round(4).to_string(index=False))

print("\n=== Cox, raw encoding (original approach) ===")
print(cox[cox.encoding == "raw"].round(4).to_string(index=False))
