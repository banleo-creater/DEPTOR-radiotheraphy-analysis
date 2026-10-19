#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GSE41613 -- formal DEPTOR x radiotherapy interaction validation.

Cohort: n=96 HPV-negative OSCC (GPL570, processed series matrix available).
RT proxy: `treatment` == "multi-modality" (surgery + RT +/- chemo) vs
          "uni-modality" (surgery only).  This is the only treatment variable
          the cohort provides; it conflates RT with chemotherapy.

Endpoints: OS (fu time, months; vital).  DSS is NOT available in this cohort.

Analyses (mirrors the TCGA predictive-vs-prognostic pipeline):
  1. median-split DEPTOR (dichotomous) x RT Cox interaction + LRT
  2. continuous DEPTOR (per SD) x RT Cox interaction (sensitivity)
  3. per-arm HR of DEPTOR-high (adjusted for stage)
  4. four-cell event rates (DEPTOR hi/lo x RT yes/no)

Outputs -> data/results_gse41613_rt/
"""
import os, sys, gzip, json
import numpy as np, pandas as pd
from lifelines import CoxPHFitter
from scipy.stats import chi2 as chi2_dist
import warnings; warnings.filterwarnings("ignore")

BASE = r"WorkBuddy/2026-10-17-11-15-46"
GEO = r"<PATH>"
COHORT = "GSE41613"
OUT = os.path.join(BASE, "data", "results_gse41613_rt")
os.makedirs(OUT, exist_ok=True)

# ------------------------------------------------------------------ IO helpers
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


def cox_fit(d, covs, t="os_t", e="os_e"):
    cph = CoxPHFitter()
    dd = d[[t, e] + covs].dropna()
    cph.fit(dd, duration_col=t, event_col=e)
    return cph, dd


# ------------------------------------------------------------------ load
print(f"===== {COHORT} : DEPTOR x RT interaction =====")
p2s = gpl570_symbols(os.path.join(BASE, "data", "plat", "GPL570.annot.gz"))
raw = parse_matrix(os.path.join(GEO, COHORT, f"{COHORT}_series_matrix.txt.gz"))
raw["symbol"] = [p2s.get(i) for i in raw.index]
raw = raw.dropna(subset=["symbol"])
expr = raw.groupby("symbol").max().T          # samples x genes (log2)
print("gene-level matrix:", expr.shape)

clin = pd.read_csv(os.path.join(BASE, "data", "gse41613_clinical.csv"), index_col=0)
clin = clin.reindex(expr.index)

d = pd.DataFrame(index=expr.index)
d["DEPTOR"] = expr["DEPTOR"]
d["os_t"] = pd.to_numeric(clin["fu time"], errors="coerce")
d["os_e"] = clin["vital"].astype(str).str.lower().str.startswith("dead").astype(int)
d["rt"] = clin["treatment"].astype(str).str.lower().str.contains("multi").astype(int)
d.loc[clin["treatment"].astype(str).str.lower().eq("unknown"), "rt"] = np.nan
d["stage_late"] = clin["tumor stage"].astype(str).str.contains("III|IV").astype(int)
d["age"] = pd.to_numeric(clin["age"], errors="coerce")   # may be banded -> NaN
d["sex"] = (clin["Sex"].astype(str).str.upper().str.startswith("M")).astype(int)

d = d.dropna(subset=["DEPTOR", "os_t", "os_e", "rt"])
print(f"analysable: n={len(d)} | events={int(d.os_e.sum())} | RT+={int(d.rt.sum())}")

# median split (cohort-internal)
med = d["DEPTOR"].median()
d["DEPTOR_hi"] = (d["DEPTOR"] > med).astype(int)
d["DEPTOR_z"] = (d["DEPTOR"] - d["DEPTOR"].mean()) / d["DEPTOR"].std()
print(f"DEPTOR median = {med:.3f}; high n={int(d.DEPTOR_hi.sum())}")

# ------------------------------------------------------------------ 1) dichotomous interaction + LRT
print("\n-- 1) dichotomous DEPTOR x RT --")
d["inter_d"] = d["DEPTOR_hi"] * d["rt"]
rows = []
for covs, tag in [
    (["DEPTOR_hi", "rt"], "M0_no_interaction"),
    (["DEPTOR_hi", "rt", "inter_d"], "M1_with_interaction"),
    (["DEPTOR_hi", "rt", "inter_d", "stage_late"], "M1_adj_stage"),
]:
    cph, dd = cox_fit(d, covs)
    s = cph.summary
    rows.append(dict(model=tag, n=len(dd), events=int(dd.os_e.sum()),
                     DEPTOR_hi_HR=round(s.loc["DEPTOR_hi", "exp(coef)"], 3),
                     DEPTOR_hi_p=round(s.loc["DEPTOR_hi", "p"], 4),
                     rt_HR=round(s.loc["rt", "exp(coef)"], 3),
                     interaction_HR=round(s.loc["inter_d", "exp(coef)"], 3) if "inter_d" in s.index else np.nan,
                     interaction_lo=round(s.loc["inter_d", "exp(coef) lower 95%"], 3) if "inter_d" in s.index else np.nan,
                     interaction_hi=round(s.loc["inter_d", "exp(coef) upper 95%"], 3) if "inter_d" in s.index else np.nan,
                     interaction_p=round(s.loc["inter_d", "p"], 4) if "inter_d" in s.index else np.nan,
                     logHR=round(s.loc["inter_d", "coef"], 4) if "inter_d" in s.index else np.nan,
                     se=round(s.loc["inter_d", "se(coef)"], 4) if "inter_d" in s.index else np.nan))
    print(f"  {tag:22s} " + (f"inter HR={s.loc['inter_d','exp(coef)']:.2f} "
          f"({s.loc['inter_d','exp(coef) lower 95%']:.2f}-{s.loc['inter_d','exp(coef) upper 95%']:.2f}) "
          f"p={s.loc['inter_d','p']:.3g}" if "inter_d" in s.index else
          f"DEPTOR_hi HR={s.loc['DEPTOR_hi','exp(coef)']:.2f}"))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "GSE41613_interaction_dichotomous.csv"), index=False)

# LRT (manual: 2*dLL, df=1)
c0, dd0 = cox_fit(d, ["DEPTOR_hi", "rt"])
c1, dd1 = cox_fit(d, ["DEPTOR_hi", "rt", "inter_d"])
ll_diff = 2.0 * (float(c1.log_likelihood_) - float(c0.log_likelihood_))
ll_diff = max(ll_diff, 0.0)
lrt_p = float(chi2_dist.sf(ll_diff, 1))
print(f"  LRT chi2={ll_diff:.2f} df=1 p={lrt_p:.3g}")
pd.DataFrame([dict(cohort=COHORT, endpoint="OS", LRT_chi2=round(ll_diff, 3),
                   LRT_p=lrt_p, n=len(dd0), events=int(dd0.os_e.sum()))]
             ).to_csv(os.path.join(OUT, "GSE41613_LRT.csv"), index=False)

# ------------------------------------------------------------------ 2) continuous sensitivity
print("\n-- 2) continuous DEPTOR (per SD) x RT --")
d["inter_z"] = d["DEPTOR_z"] * d["rt"]
cph, dd = cox_fit(d, ["DEPTOR_z", "rt", "inter_z"])
s = cph.summary
print(f"  inter HR={s.loc['inter_z','exp(coef)']:.2f} "
      f"({s.loc['inter_z','exp(coef) lower 95%']:.2f}-{s.loc['inter_z','exp(coef) upper 95%']:.2f}) "
      f"p={s.loc['inter_z','p']:.3g}")
pd.DataFrame([dict(cohort=COHORT, endpoint="OS",
                   inter_HR=round(s.loc["inter_z", "exp(coef)"], 3),
                   lo=round(s.loc["inter_z", "exp(coef) lower 95%"], 3),
                   hi=round(s.loc["inter_z", "exp(coef) upper 95%"], 3),
                   p=round(s.loc["inter_z", "p"], 4),
                   logHR=round(s.loc["inter_z", "coef"], 4),
                   se=round(s.loc["inter_z", "se(coef)"], 4),
                   n=len(dd), events=int(dd.os_e.sum()))]
             ).to_csv(os.path.join(OUT, "GSE41613_interaction_continuous.csv"), index=False)

# ------------------------------------------------------------------ 3) per-arm HR
print("\n-- 3) DEPTOR-high HR within each arm (adj. stage) --")
arows = []
for arm, lbl, n in [(0, "no RT (uni-modality)", int((d.rt == 0).sum())),
                    (1, "RT (multi-modality)", int((d.rt == 1).sum()))]:
    sub = d[d.rt == arm]
    cph, dd = cox_fit(sub, ["DEPTOR_hi", "stage_late"])
    s = cph.summary.loc["DEPTOR_hi"]
    arows.append(dict(cohort=COHORT, endpoint="OS", arm=lbl,
                      HR=round(s["exp(coef)"], 3),
                      lo=round(s["exp(coef) lower 95%"], 3),
                      hi=round(s["exp(coef) upper 95%"], 3),
                      p=round(s["p"], 4), n=len(dd), events=int(dd.os_e.sum())))
    print(f"  {lbl:24s} HR={s['exp(coef)']:.2f} "
          f"({s['exp(coef) lower 95%']:.2f}-{s['exp(coef) upper 95%']:.2f}) "
          f"p={s['p']:.3g} n={len(dd)} ev={int(dd.os_e.sum())}")
pd.DataFrame(arows).to_csv(os.path.join(OUT, "GSE41613_arm_HR.csv"), index=False)

# ------------------------------------------------------------------ 4) four-cell event rates
print("\n-- 4) four-cell event rates (OS) --")
fr = []
for hi in (0, 1):
    for rt in (0, 1):
        sub = d[(d.DEPTOR_hi == hi) & (d.rt == rt)]
        fr.append(dict(cohort=COHORT, endpoint="OS", DEPTOR_hi=hi, rt=rt,
                       n=len(sub), events=int(sub.os_e.sum()),
                       rate=round(float(sub.os_e.mean()), 4) if len(sub) else np.nan))
four = pd.DataFrame(fr)
print(four.to_string(index=False))
four.to_csv(os.path.join(OUT, "GSE41613_four_cell_event_rate.csv"), index=False)

print(f"\nDONE -> {OUT}")
