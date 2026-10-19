#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TCGA-HNSC — DEPTOR : PREDICTIVE vs PROGNOSTIC formal statistical argument
=========================================================================
Builds the discrimination + clinical-utility evidence that clinical journals
want, entirely from the existing TCGA master table:

  1. Time-dependent AUC(t)  (Uno's IPCW estimator, KM censoring weights)
        - Model 0 : p16 (HPV) + AJCC stage            (clinical baseline)
        - Model 1 : Model 0 + DEPTOR (continuous, z)   (molecular add-on)
        - at t = 12, 24, 36, 60 months, with bootstrap 95% CI + ΔAUC(t)
  2. C-index increment (ΔC-index)
        - Harrell's C for M0 vs M1, bootstrap 95% CI on the difference
        - decomposed into RT=Yes arm and RT=No arm to split
          PROGNOSTIC (both arms) vs PREDICTIVE (interaction-driven) signal
  3. Decision Curve Analysis (3-year DSS)
        - net benefit of M0 vs M1 vs treat-all vs treat-none
        - over threshold probability grid
  4. Formal predictive/prognostic separation
        - Likelihood-ratio test of DEPTOR x RT interaction  (PREDICTIVE)
        - Adjusted main effect of DEPTOR (p16+stage+RT)    (PROGNOSTIC)

Primary endpoint : DSS (disease-specific survival)
Confirmatory     : OS
"""
import os, json, warnings, time as _t
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy import stats
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter

BASE = "WorkBuddy/2026-10-17-11-15-46"
OUT  = os.path.join(BASE, "data", "results_ppstats"); os.makedirs(OUT, exist_ok=True)
np.random.seed(20261018)

# ----------------------------------------------------------------------------
# 1. DATA
# ----------------------------------------------------------------------------
raw = pd.read_csv(os.path.join(BASE, "data", "tcga_hnsc_pca_deptor.csv"))
def ev(s): return s.astype(str).str.startswith("1").astype(int)
d = pd.DataFrame()
d["id"]      = raw["patientId"]
d["DEPTOR"]  = pd.to_numeric(raw["DEPTOR_RSEM"], errors="coerce")
# biomarker z-score (continuous)
d["DEPTOR_z"]= (d["DEPTOR"]-d["DEPTOR"].mean())/d["DEPTOR"].std()
d["DEPTOR_hi"]= (d["DEPTOR"] >= d["DEPTOR"].median()).astype(int)   # high vs low
# p16 / HPV (SUBTYPE)
d["p16pos"]  = raw["SUBTYPE"].map(lambda s: 1 if str(s).endswith("HPV+") else 0)
# stage : late = III / IV  vs  early = I / II
stg = raw["AJCC_PATHOLOGIC_TUMOR_STAGE"].astype(str).str.upper()
d["stage_late"] = stg.map(lambda s: 1 if ("III" in s) or ("IV" in s) else 0)
d["RT"]      = raw["RADIATION_THERAPY"].astype(str).str.strip().str.title()
d["RT_yes"]  = (d["RT"]=="Yes").astype(int)
d["AGE"]     = pd.to_numeric(raw["AGE"], errors="coerce")
d["SEX_M"]   = (raw["SEX"].astype(str).str.upper().str.startswith("M")).astype(int)
# endpoints
for ep,(tc,sc) in [("DSS",("DSS_MONTHS","DSS_STATUS")),("OS",("OS_MONTHS","OS_STATUS"))]:
    d[f"{ep}_T"] = pd.to_numeric(raw[tc], errors="coerce")
    d[f"{ep}_E"] = ev(raw[sc])
print(f"[data] n={len(d)}  | RT Yes={int(d.RT_yes.sum())} No={int((d.RT_yes==0).sum())} "
      f"| p16+={int(d.p16pos.sum())}  | stage_late={int(d.stage_late.sum())} "
      f"| DSS events={int(d.DSS_E.sum())}  | DEPTOR median={d.DEPTOR.median():.1f}")

EP = "DSS"                  # primary endpoint
Tcol, Ecol = f"{EP}_T", f"{EP}_E"
TIMES = [12, 24, 36, 60]    # months

# ----------------------------------------------------------------------------
# 2. COX helpers
# ----------------------------------------------------------------------------
def fit_cox(df, cols, tcol=Tcol, ecol=Ecol):
    dd = df.dropna(subset=[tcol, ecol]+cols).copy()
    dd = dd[dd[tcol] > 0]
    if dd[ecol].sum() < 5 or dd.shape[0] < 20: return None
    X = dd[cols].copy()
    cph = CoxPHFitter()
    cph.fit(X.assign(**{tcol: dd[tcol], ecol: dd[ecol]}),
            duration_col=tcol, event_col=ecol)
    return cph, dd

def predict_survival_at(cph, X, t):
    """S(t|x) from Cox baseline cumulative hazard * partial hazard"""
    ph = cph.predict_partial_hazard(X).values.ravel()
    bch = cph.baseline_cumulative_hazard_
    ts  = bch.index.values.astype(float)
    h0  = bch.iloc[:,0].values.astype(float)
    # cumulative hazard at t (step/linear interp on cumulative)
    if t <= ts[0]:
        H0 = 0.0
    elif t >= ts[-1]:
        H0 = h0[-1]
    else:
        H0 = np.interp(t, ts, h0)
    return np.exp(-H0 * ph)

# ----------------------------------------------------------------------------
# 3. Uno IPCW time-dependent AUC  (O(n log n))
# ----------------------------------------------------------------------------
def km_censoring_survival(T, cens_ind, eval_t):
    """KM of censoring distribution G(t)=P(C>t). cens_ind: 1 if censored.
    At each censoring time tau, at-risk = #subjects with observed time >= tau."""
    t = np.asarray(T, float); c = np.asarray(cens_ind, int)
    cet = np.sort(np.unique(t[c == 1])) if (c == 1).any() else np.array([])
    surv_after = []; s = 1.0
    for tau in cet:
        n = int(np.sum(t >= tau)); d = int(np.sum((t == tau) & (c == 1)))
        if n > 0 and d > 0:
            s *= (n - d) / n
        surv_after.append(s)
    out = np.empty(len(eval_t))
    for i, te in enumerate(eval_t):
        idx = int(np.searchsorted(cet, te, "right")) - 1
        out[i] = 1.0 if idx < 0 else surv_after[idx]
    return np.clip(out, 1e-3, 1.0)

def uno_auc(T, D, eta, cens_ind, times):
    """Uno's IPCW time-dependent AUC at each t (O(n log n)), bounded [0,1].
    Cases are weighted by Ĝ(T_i)^{-1} at their actual event time T_i.
    Tied risk scores count as 0.5 (handles discrete models correctly)."""
    res = {}
    for ti in times:
        case = (T <= ti) & (D == 1)
        ctrl = (T >  ti)
        if case.sum() == 0 or ctrl.sum() == 0:
            res[ti] = np.nan; continue
        C = int(ctrl.sum())
        ec = eta[case]
        Gi = km_censoring_survival(T, cens_ind, T[case])   # Ĝ(T_i) per case
        wc = 1.0 / np.clip(Gi, 1e-3, 1.0)
        sc = np.sort(eta[ctrl])
        num = 0.0; den = 0.0
        for k in range(len(ec)):
            lo = np.searchsorted(sc, ec[k], side="left")
            hi = np.searchsorted(sc, ec[k], side="right")
            eq = hi - lo
            num += wc[k] * (lo + 0.5 * eq)
            den += wc[k] * C
        res[ti] = num / den if den > 0 else np.nan
    return res

def roc_metrics(df, cols):
    """return T,D,cens_ind,eta for AUC"""
    dd = df.dropna(subset=[Tcol, Ecol]+cols).copy(); dd = dd[dd[Tcol]>0]
    cph, _ = fit_cox(dd, cols)
    if cph is None: return None
    eta = cph.predict_partial_hazard(dd[cols]).values.ravel()
    T = dd[Tcol].values.astype(float); D = dd[Ecol].values.astype(int)
    cens_ind = 1 - D
    return T, D, cens_ind, eta, dd

# ----------------------------------------------------------------------------
# 4. BOOTSTRAP point estimates
# ----------------------------------------------------------------------------
def boot_indices(n, B):
    for _ in range(B):
        yield np.random.randint(0, n, n)

B_AUC, B_C = 500, 1000

# ---- model definitions ----
M0 = ["p16pos", "stage_late"]              # clinical baseline (p16 + stage)
M1 = M0 + ["DEPTOR_z"]                      # + DEPTOR

# full-data AUC
r0 = roc_metrics(d, M0); r1 = roc_metrics(d, M1)
auc0 = uno_auc(r0[0], r0[1], r0[3], r0[2], TIMES)
auc1 = uno_auc(r1[0], r1[1], r1[3], r1[2], TIMES)
print("\n[time-dependent AUC]  Uno IPCW")
print(f"  {'t(mo)':>6} {'AUC M0':>9} {'AUC M1':>9} {'ΔAUC':>8}")
for t in TIMES:
    print(f"  {t:>6} {auc0[t]:>9.3f} {auc1[t]:>9.3f} {auc1[t]-auc0[t]:>+8.3f}")

# bootstrap AUC CI (collect paired M0/M1 within the same resample)
auc0_b = {t:[] for t in TIMES}; auc1_b = {t:[] for t in TIMES}
n_full = len(d)
for idx in boot_indices(n_full, B_AUC):
    b = d.iloc[idx]
    rb0 = roc_metrics(b, M0); rb1 = roc_metrics(b, M1)
    if rb0 is None or rb1 is None: continue
    a0 = uno_auc(rb0[0], rb0[1], rb0[3], rb0[2], TIMES)
    a1 = uno_auc(rb1[0], rb1[1], rb1[3], rb1[2], TIMES)
    if any(np.isnan(a0[t]) or np.isnan(a1[t]) for t in TIMES): continue
    for t in TIMES:
        auc0_b[t].append(a0[t]); auc1_b[t].append(a1[t])
auc0_lo = {t: np.percentile(auc0_b[t],2.5) for t in TIMES}
auc0_hi = {t: np.percentile(auc0_b[t],97.5) for t in TIMES}
auc1_lo = {t: np.percentile(auc1_b[t],2.5) for t in TIMES}
auc1_hi = {t: np.percentile(auc1_b[t],97.5) for t in TIMES}
dAUC = {t: auc1[t]-auc0[t] for t in TIMES}
dAUC_lo = {t: np.percentile(np.array(auc1_b[t])-np.array(auc0_b[t]),2.5) for t in TIMES}
dAUC_hi = {t: np.percentile(np.array(auc1_b[t])-np.array(auc0_b[t]),97.5) for t in TIMES}
print("  bootstrap 95% CI on ΔAUC:", {t:(round(dAUC_lo[t],3),round(dAUC_hi[t],3)) for t in TIMES})

# ----------------------------------------------------------------------------
# 5. C-index + ΔC-index (bootstrap)  — overall and by arm
# ----------------------------------------------------------------------------
def cindex_of(df, cols):
    dd = df.dropna(subset=[Tcol, Ecol]+cols).copy(); dd = dd[dd[Tcol]>0]
    cph, _ = fit_cox(dd, cols)
    if cph is None: return None
    eta = cph.predict_partial_hazard(dd[cols]).values.ravel()
    return concordance_index(dd[Tcol].values, -eta, dd[Ecol].values)

def boot_delta_c(df, b0, b1, B):
    n = len(df); out = []
    for idx in boot_indices(n, B):
        b = df.iloc[idx]
        c0 = cindex_of(b, b0); c1 = cindex_of(b, b1)
        if c0 is None or c1 is None: continue
        out.append(c1-c0)
    return np.array(out)

# overall
c0_all = cindex_of(d, M0); c1_all = cindex_of(d, M1)
dC_all = c1_all - c0_all
dC_all_b = boot_delta_c(d, M0, M1, B_C)
dC_all_lo, dC_all_hi = np.percentile(dC_all_b, [2.5,97.5])
print(f"\n[C-index]  overall  C(M0)={c0_all:.3f}  C(M1)={c1_all:.3f}  ΔC={dC_all:+.4f} "
      f"(95% CI {dC_all_lo:+.4f}..{dC_all_hi:+.4f})")

# by arm (prognostic vs predictive decomposition)
arm_rows = []
for arm, mask in [("RT=Yes", d.RT_yes==1), ("RT=No", d.RT_yes==0)]:
    sub = d[mask]
    c0 = cindex_of(sub, M0); c1 = cindex_of(sub, M1)
    if c0 is None or c1 is None: continue
    dc = c1-c0
    dcb = boot_delta_c(sub, M0, M1, 600)
    lo, hi = np.percentile(dcb,[2.5,97.5]) if len(dcb) else (np.nan,np.nan)
    arm_rows.append(dict(arm=arm, n=len(sub), C_M0=round(c0,3), C_M1=round(c1,3),
                         dC=round(dc,4), dC_lo=round(lo,4), dC_hi=round(hi,4)))
    print(f"  {arm:7s} n={len(sub):3d}  C(M0)={c0:.3f} C(M1)={c1:.3f} ΔC={dc:+.4f} "
          f"(95% CI {lo:+.4f}..{hi:+.4f})")
arm_df = pd.DataFrame(arm_rows)

# ----------------------------------------------------------------------------
# 6. DECISION CURVE ANALYSIS (3-year DSS)
# ----------------------------------------------------------------------------
T_DCA = 36
def net_benefit(risk, event, pt):
    treat = (risk >= pt).astype(int)
    tp = ((treat==1) & (event==1)).sum(); fp = ((treat==1) & (event==0)).sum()
    N = len(risk)
    return (tp - fp*pt/(1-pt))/N

cph0, dd0 = fit_cox(d, M0); cph1, dd1 = fit_cox(d, M1)
# align risk to full-data rows that have DSS
risk0 = 1 - predict_survival_at(cph0, dd0[M0], T_DCA)
risk1 = 1 - predict_survival_at(cph1, dd1[M1], T_DCA)
ev0 = dd0[Ecol].values.astype(int)
# prevalence (event within 3y among those with data)
prev = ev0.mean()
pts = np.round(np.arange(0.05, 0.505, 0.05), 2)
dca = pd.DataFrame({"threshold": pts,
                    "treat_all": prev - (1-prev)*pts/(1-pts),
                    "M0": [net_benefit(risk0, ev0, p) for p in pts],
                    "M1": [net_benefit(risk1, ev0, p) for p in pts]})
# net benefit gain (M1 - M0) at each threshold
dca["NB_gain_M1_over_M0"] = dca["M1"] - dca["M0"]
# bootstrap CI of NB gain at a few key thresholds
NB_KEY = [0.10, 0.20, 0.30]
nb_gain_b = {p:[] for p in NB_KEY}
for _ in range(B_AUC):
    idx = np.random.randint(0, len(d), len(d))
    b = d.iloc[idx]
    cb0, bb0 = fit_cox(b, M0); cb1, bb1 = fit_cox(b, M1)
    if cb0 is None or cb1 is None: continue
    r0b = 1-predict_survival_at(cb0, bb0[M0], T_DCA)
    r1b = 1-predict_survival_at(cb1, bb1[M1], T_DCA)
    eb = bb0[Ecol].values.astype(int)
    for p in NB_KEY:
        nb_gain_b[p].append(net_benefit(r1b, eb, p) - net_benefit(r0b, eb, p))
print(f"\n[DCA] 3-year DSS  prevalence_event_within_3y={prev:.3f}")
print("  threshold  NB(M0)  NB(M1)  gain   gain 95% CI")
nb_ci = {}
for p in NB_KEY:
    g = np.array(nb_gain_b[p]); lo,hi = np.percentile(g,[2.5,97.5])
    nb_ci[p] = (lo,hi)
    row = dca.iloc[(dca.threshold-p).abs().idxmin()]
    print(f"    {p:.2f}    {row.M0:.4f}  {row.M1:.4f}  {row.NB_gain_M1_over_M0:+.4f}   "
          f"({lo:+.4f}..{hi:+.4f})")

# ----------------------------------------------------------------------------
# 7. PREDICTIVE (interaction LRT) vs PROGNOSTIC (main effect)
# ----------------------------------------------------------------------------
def lrt_with_interaction(df, continuous=True):
    dd = df.dropna(subset=[Tcol,Ecol,"p16pos","stage_late","RT_yes","DEPTOR_z"]).copy()
    dd = dd[dd[Tcol]>0]
    base_cols = ["p16pos","stage_late","RT_yes"]
    if continuous:
        term = "DEPTOR_z"
    else:
        dd = dd.assign(DEPTOR_hi=dd["DEPTOR_hi"])
        term = "DEPTOR_hi"
    full = dd[base_cols+[term]].copy(); full["INT"] = (dd[term]*dd["RT_yes"]).values
    full2= full[base_cols+[term]+["INT"]]
    cph_r = CoxPHFitter().fit(full[base_cols+[term]].assign(**{Tcol:dd[Tcol],Ecol:dd[Ecol]}),
                              duration_col=Tcol, event_col=Ecol)
    cph_f = CoxPHFitter().fit(full2.assign(**{Tcol:dd[Tcol],Ecol:dd[Ecol]}),
                              duration_col=Tcol, event_col=Ecol)
    lr = 2*(cph_f.log_likelihood_ - cph_r.log_likelihood_)
    p = stats.chi2.sf(lr, 1)
    hr_int = float(np.exp(cph_f.params_["INT"])); p_int = float(cph_f.summary.loc["INT","p"])
    # main effect (prognostic) from reduced model
    hr_main = float(np.exp(cph_r.params_[term])); p_main = float(cph_r.summary.loc[term,"p"])
    return dict(continuous=continuous, n=len(dd), LRT_chi2=round(lr,3), LRT_p=round(p,5),
                INT_HR=round(hr_int,3), INT_p=round(p_int,5),
                main_HR=round(hr_main,3), main_p=round(p_main,4))

lrt_cont = lrt_with_interaction(d, continuous=True)
lrt_dich = lrt_with_interaction(d, continuous=False)
print("\n[predictive vs prognostic]")
print("  continuous DEPTOR : interaction LRT chi2=%s p=%s | INT HR=%s p=%s | main HR=%s p=%s"
      % (lrt_cont["LRT_chi2"], lrt_cont["LRT_p"], lrt_cont["INT_HR"], lrt_cont["INT_p"],
         lrt_cont["main_HR"], lrt_cont["main_p"]))
print("  dichot DEPTOR     : interaction LRT chi2=%s p=%s | INT HR=%s p=%s | main HR=%s p=%s"
      % (lrt_dich["LRT_chi2"], lrt_dich["LRT_p"], lrt_dich["INT_HR"], lrt_dich["INT_p"],
         lrt_dich["main_HR"], lrt_dich["main_p"]))

# interaction HR per endpoint (for forest)
int_rows = []
for ep in ["DSS","OS"]:
    tc,ec = f"{ep}_T", f"{ep}_E"
    dd = d.dropna(subset=[tc,ec,"DEPTOR_hi","RT_yes"]).copy(); dd=dd[dd[tc]>0]
    dd["INT"]=dd["DEPTOR_hi"]*dd["RT_yes"]
    cph = CoxPHFitter().fit(dd[["DEPTOR_hi","RT_yes","INT"]].assign(**{tc:dd[tc],ec:dd[ec]}),
                            duration_col=tc, event_col=ec)
    int_rows.append(dict(endpoint=ep, n=len(dd), INT_HR=round(float(np.exp(cph.params_["INT"])),3),
                        INT_lo=round(float(np.exp(cph.confidence_intervals_.loc["INT"]).values[0]),3),
                        INT_hi=round(float(np.exp(cph.confidence_intervals_.loc["INT"]).values[1]),3),
                        INT_p=round(float(cph.summary.loc["INT","p"]),5)))
int_df = pd.DataFrame(int_rows)

# ----------------------------------------------------------------------------
# 7b. DICHOTOMOUS DEPTOR model (clinically actionable high/low cut)
#     Because the predictive signal is dichotomisation-driven, redo the same
#     discrimination/utility metrics with DEPTOR entered as high vs low.
# ----------------------------------------------------------------------------
M1d = M0 + ["DEPTOR_hi"]
r1d = roc_metrics(d, M1d)
auc1d = uno_auc(r1d[0], r1d[1], r1d[3], r1d[2], TIMES)
auc1d_b = {t:[] for t in TIMES}
for idx in boot_indices(n_full, B_AUC):
    b = d.iloc[idx]
    rb1d = roc_metrics(b, M1d)
    if rb1d is None: continue
    a = uno_auc(rb1d[0], rb1d[1], rb1d[3], rb1d[2], TIMES)
    for t in TIMES:
        if not np.isnan(a[t]): auc1d_b[t].append(a[t])
auc1d_lo = {t: np.percentile(auc1d_b[t],2.5) for t in TIMES}
auc1d_hi = {t: np.percentile(auc1d_b[t],97.5) for t in TIMES}
print("\n[time-dependent AUC] M1d (dichotomous DEPTOR)")
print(f"  t(mo)  AUC M0   AUC M1d   ΔAUC")
for t in TIMES:
    print(f"  {t:>5} {auc0[t]:>8.3f} {auc1d[t]:>8.3f} {auc1d[t]-auc0[t]:>+8.3f}")

# C-index dichotomous, overall + by arm
c1d_all = cindex_of(d, M1d); dC1d_all = c1d_all - c0_all
dC1d_b = boot_delta_c(d, M0, M1d, B_C); dC1d_lo, dC1d_hi = np.percentile(dC1d_b,[2.5,97.5])
arm_rows_d = []
for arm, mask in [("RT=Yes", d.RT_yes==1), ("RT=No", d.RT_yes==0)]:
    sub = d[mask]; c0a=cindex_of(sub,M0); c1a=cindex_of(sub,M1d)
    if c0a is None or c1a is None: continue
    dc=c1a-c0a; dcb=boot_delta_c(sub,M0,M1d,600); lo,hi=np.percentile(dcb,[2.5,97.5]) if len(dcb) else (np.nan,np.nan)
    arm_rows_d.append(dict(model="M1d", arm=arm, n=len(sub), C_M0=round(c0a,3), C_M1=round(c1a,3),
                           dC=round(dc,4), dC_lo=round(lo,4), dC_hi=round(hi,4)))
arm_df2 = pd.DataFrame(arm_rows_d)
print(f"[C-index] M1d overall ΔC={dC1d_all:+.4f} (95% CI {dC1d_lo:+.4f}..{dC1d_hi:+.4f})")

# DCA dichotomous
cph1d, dd1d = fit_cox(d, M1d)
risk1d = 1 - predict_survival_at(cph1d, dd1d[M1d], T_DCA)
ev1d = dd1d[Ecol].values.astype(int)
dca["M1d"] = [net_benefit(risk1d, ev1d, p) for p in pts]
dca["NB_gain_M1d_over_M0"] = dca["M1d"] - dca["M0"]
print(f"[DCA] M1d vs M0 max gain at pt=0.20: {dca.loc[dca.threshold==0.20,'NB_gain_M1d_over_M0'].values[0]:+.4f}")

# ----------------------------------------------------------------------------
# 7c. ARM-SPECIFIC DEPTOR EFFECT (qualitative interaction)
#     If DEPTOR's effect REVERSES across arms, that is the strongest form of
#     a predictive (treatment-modifying) biomarker.
# ----------------------------------------------------------------------------
arm_eff = []
for arm, m in [("RT=No", d.RT_yes==0), ("RT=Yes", d.RT_yes==1)]:
    sub = d[m].dropna(subset=[Tcol,Ecol,"p16pos","stage_late","DEPTOR_hi"]); sub=sub[sub[Tcol]>0]
    cph = CoxPHFitter().fit(sub[["p16pos","stage_late","DEPTOR_hi"]].assign(**{Tcol:sub[Tcol],Ecol:sub[Ecol]}),
                            duration_col=Tcol, event_col=Ecol)
    lo,hi = cph.confidence_intervals_.loc["DEPTOR_hi"].values
    arm_eff.append(dict(arm=arm, n=len(sub), events=int(sub[Ecol].sum()),
                        HR=round(float(np.exp(cph.params_["DEPTOR_hi"])),3),
                        lo=round(float(np.exp(lo)),3), hi=round(float(np.exp(hi)),3),
                        p=round(float(cph.summary.loc["DEPTOR_hi","p"]),5)))
arm_eff_df = pd.DataFrame(arm_eff)
print("\n[arm-specific DEPTOR-high effect] (prognostic in untreated, protective if irradiated)")
print(arm_eff_df.to_string(index=False))
arm_eff_df.to_csv(os.path.join(OUT,"arm_specific_effect.csv"), index=False)

# 4-cell DSS event rates
cells = d.groupby(["DEPTOR_hi","RT_yes"]).agg(n=(Ecol,"size"), ev=(Ecol,"sum"))
cells["rate"] = cells["ev"]/cells["n"]
cells.reset_index().to_csv(os.path.join(OUT,"four_cell_event_rate.csv"), index=False)

# ----------------------------------------------------------------------------
# 8. SAVE TABLES
# ----------------------------------------------------------------------------
auc_tbl = pd.DataFrame({"time_months":TIMES,
    "AUC_M0":[round(auc0[t],3) for t in TIMES],
    "AUC_M0_lo":[round(auc0_lo[t],3) for t in TIMES],
    "AUC_M0_hi":[round(auc0_hi[t],3) for t in TIMES],
    "AUC_M1_cont":[round(auc1[t],3) for t in TIMES],
    "AUC_M1_cont_lo":[round(auc1_lo[t],3) for t in TIMES],
    "AUC_M1_cont_hi":[round(auc1_hi[t],3) for t in TIMES],
    "delta_AUC_cont":[round(dAUC[t],3) for t in TIMES],
    "delta_AUC_cont_lo":[round(dAUC_lo[t],3) for t in TIMES],
    "delta_AUC_cont_hi":[round(dAUC_hi[t],3) for t in TIMES],
    "AUC_M1_dich":[round(auc1d[t],3) for t in TIMES],
    "AUC_M1_dich_lo":[round(auc1d_lo[t],3) for t in TIMES],
    "AUC_M1_dich_hi":[round(auc1d_hi[t],3) for t in TIMES],
    "delta_AUC_dich":[round(auc1d[t]-auc0[t],3) for t in TIMES]})
auc_tbl.to_csv(os.path.join(OUT,"tdAUC_DSS.csv"), index=False)

cidx_tbl = pd.DataFrame([
    dict(model="M0 (p16+stage)", C=round(c0_all,4), note="baseline"),
    dict(model="M1 (p16+stage+DEPTOR cont)", C=round(c1_all,4), note="continuous add-on"),
    dict(model="ΔC cont (M1-M0)", C=round(dC_all,4),
         note=f"95% CI {dC_all_lo:+.4f}..{dC_all_hi:+.4f}"),
    dict(model="M1d (p16+stage+DEPTOR hi/lo)", C=round(c1d_all,4), note="dichotomous add-on"),
    dict(model="ΔC dich (M1d-M0)", C=round(dC1d_all,4),
         note=f"95% CI {dC1d_lo:+.4f}..{dC1d_hi:+.4f}")])
cidx_tbl.to_csv(os.path.join(OUT,"cindex_overall.csv"), index=False)
arm_all = pd.concat([arm_df.assign(model="M1_cont"), arm_df2], ignore_index=True)
arm_all.to_csv(os.path.join(OUT,"cindex_by_arm.csv"), index=False)
dca.to_csv(os.path.join(OUT,"dca_3yr_DSS.csv"), index=False)
pd.DataFrame([lrt_cont,lrt_dich]).to_csv(os.path.join(OUT,"interaction_LRT.csv"), index=False)
int_df.to_csv(os.path.join(OUT,"interaction_HR_forest.csv"), index=False)

# ----------------------------------------------------------------------------
# 9. FIGURES
# ----------------------------------------------------------------------------
plt.rcParams.update({"font.size":11,"axes.grid":True,"grid.alpha":.25,
                     "figure.dpi":160,"savefig.dpi":160})

# Fig 1: time-dependent AUC M0 vs M1 (continuous) vs M1d (dichotomous)
fig, ax = plt.subplots(figsize=(6.8,4.6))
ax.fill_between(TIMES, [auc0_lo[t] for t in TIMES], [auc0_hi[t] for t in TIMES],
                color="#7aa6c2", alpha=.20, label="M0 95% CI")
ax.fill_between(TIMES, [auc1_lo[t] for t in TIMES], [auc1_hi[t] for t in TIMES],
                color="#e6550d", alpha=.20, label="M1(cont) 95% CI")
ax.fill_between(TIMES, [auc1d_lo[t] for t in TIMES], [auc1d_hi[t] for t in TIMES],
                color="#31a354", alpha=.20, label="M1d(dich) 95% CI")
ax.plot(TIMES, [auc0[t] for t in TIMES], "o-", color="#3182bd", lw=2, ms=6, label="M0: p16 + stage")
ax.plot(TIMES, [auc1[t] for t in TIMES], "s-", color="#e6550d", lw=2, ms=6, label="M1: + DEPTOR (continuous)")
ax.plot(TIMES, [auc1d[t] for t in TIMES], "^--", color="#31a354", lw=2, ms=6, label="M1d: + DEPTOR (high/low)")
ax.axhline(0.5, color="grey", ls="--", lw=1)
ax.set_xlabel("Time (months)"); ax.set_ylabel("Time-dependent AUC (Uno IPCW)")
ax.set_title(f"Discrimination of {EP} risk: adding DEPTOR to p16+stage")
ax.set_ylim(0.45,0.85); ax.legend(loc="lower right", fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(OUT,"F1_tdAUC.png")); plt.close()

# Fig 2: DCA
fig, ax = plt.subplots(figsize=(6.8,4.6))
ax.plot(dca.threshold, dca["treat_all"], "--", color="black", lw=1.6, label="Treat all")
ax.plot(dca.threshold, np.zeros_like(dca.threshold), ":", color="grey", lw=1.4, label="Treat none")
ax.plot(dca.threshold, dca["M0"], "-", color="#3182bd", lw=2, label="M0: p16 + stage")
ax.plot(dca.threshold, dca["M1"], "-", color="#e6550d", lw=2, label="M1: + DEPTOR (cont)")
ax.plot(dca.threshold, dca["M1d"], "-.", color="#31a354", lw=2, label="M1d: + DEPTOR (hi/lo)")
ax.fill_between(dca.threshold, dca["M0"], dca["M1d"], color="#31a354", alpha=.12)
ax.set_xlabel("Threshold probability (risk of 3-yr DSS event)")
ax.set_ylabel("Net benefit")
ax.set_title("Decision curve analysis — 3-year DSS")
ax.legend(loc="upper right", fontsize=9); ax.set_xlim(0.05,0.5)
ax.text(0.06, -0.22, "M0 / M1 / M1d curves coincid<PATH>",
        fontsize=8.5, color="#444", va="top",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#bbb", alpha=.9))
plt.tight_layout(); plt.savefig(os.path.join(OUT,"F2_DCA.png")); plt.close()

# Fig 3: ΔC by arm (prognostic vs predictive decomposition) — cont & dich
fig, ax = plt.subplots(figsize=(6.4,4.0))
arms = ["RT=Yes","RT=No","Overall"]
cont_d = {"RT=Yes":arm_df.set_index("arm").loc["RT=Yes","dC"],
          "RT=No":arm_df.set_index("arm").loc["RT=No","dC"], "Overall":dC_all}
dich_d = {"RT=Yes":arm_df2.set_index("arm").loc["RT=Yes","dC"],
          "RT=No":arm_df2.set_index("arm").loc["RT=No","dC"], "Overall":dC1d_all}
cont_l = {"RT=Yes":arm_df.set_index("arm").loc["RT=Yes","dC_lo"],"RT=No":arm_df.set_index("arm").loc["RT=No","dC_lo"],"Overall":dC_all_lo}
cont_h = {"RT=Yes":arm_df.set_index("arm").loc["RT=Yes","dC_hi"],"RT=No":arm_df.set_index("arm").loc["RT=No","dC_hi"],"Overall":dC_all_hi}
dich_l = {"RT=Yes":arm_df2.set_index("arm").loc["RT=Yes","dC_lo"],"RT=No":arm_df2.set_index("arm").loc["RT=No","dC_lo"],"Overall":dC1d_lo}
dich_h = {"RT=Yes":arm_df2.set_index("arm").loc["RT=Yes","dC_hi"],"RT=No":arm_df2.set_index("arm").loc["RT=No","dC_hi"],"Overall":dC1d_hi}
x = np.arange(len(arms)); w=0.32
def _err(ax, vals, los, his, xs, col, lab):
    err=[np.array(vals)-np.array(los), np.array(his)-np.array(vals)]
    ax.bar(xs, vals, w, color=col, alpha=.85, label=lab, yerr=err, capsize=4, ecolor=col)
_err(ax, [cont_d[a] for a in arms], [cont_l[a] for a in arms], [cont_h[a] for a in arms], x-w/2, "#e6550d", "ΔC (continuous)")
_err(ax, [dich_d[a] for a in arms], [dich_l[a] for a in arms], [dich_h[a] for a in arms], x+w/2, "#31a354", "ΔC (high/low)")
for xi,v in zip(x-w/2,[cont_d[a] for a in arms]): ax.text(xi,v, f"{v:+.3f}", ha="center", va="bottom", fontsize=7)
for xi,v in zip(x+w/2,[dich_d[a] for a in arms]): ax.text(xi,v, f"{v:+.3f}", ha="center", va="bottom", fontsize=7)
ax.axhline(0, color="red", ls="--", lw=1)
ax.set_xticks(x); ax.set_xticklabels(arms)
ax.set_ylabel("ΔC-index (model with DEPTOR − p16+stage)")
ax.set_title("Where does DEPTOR add discrimination?\n(largest gain in the irradiated arm ⇒ predictive component)")
ax.legend(fontsize=8); ax.set_ylim(-0.10, 0.15)
plt.tight_layout(); plt.savefig(os.path.join(OUT,"F3_deltaC_by_arm.png")); plt.close()

# Fig 4: interaction HR forest (predictive)
fig, ax = plt.subplots(figsize=(6.2,2.6))
y = np.arange(len(int_df))[::-1]
hr=int_df.INT_HR.values; lo=int_df.INT_lo.values; hi=int_df.INT_hi.values
ax.errorbar(hr, y, xerr=[hr-lo, hi-hr], fmt="o", color="#756bb1", ecolor="#756bb1", capsize=4, ms=7)
for xi,yi,v in zip(hr,y,hr):
    ax.annotate(f"{v:.2f}", (xi,yi), textcoords="offset points", xytext=(10,0),
                ha="left", va="center", fontsize=9)
ax.axvline(1, color="red", ls="--", lw=1)
ax.set_yticks(y); ax.set_yticklabels([f"{r.endpoint} (n={r.n})" for r in int_df.itertuples()])
ax.set_xscale("log"); ax.set_xlim(0.05,3)
ax.xaxis.set_major_locator(FixedLocator([0.1,0.2,0.5,1,2]))
ax.xaxis.set_major_formatter(FixedFormatter(["0.1","0.2","0.5","1","2"]))
ax.set_xlabel("HR for DEPTOR(high) × radiotherapy interaction")
ax.set_title("Predictive effect: radiotherapy benefit in DEPTOR-high")
plt.tight_layout(); plt.savefig(os.path.join(OUT,"F4_interaction_forest.png")); plt.close()

# Fig 5: 4-cell DSS event rate — qualitative interaction
fig, ax = plt.subplots(figsize=(6.4,4.2))
x = np.arange(2); w = 0.36
for lab, dy, col in [("RT = No", -w/2, "#9ecae1"), ("RT = Yes", w/2, "#fdae6b")]:
    rates=[]; ns=[]; errs=[]
    rtyes = 1 if lab=="RT = Yes" else 0
    for dh in [0,1]:
        r = cells.loc[(dh, rtyes)]
        p = r["rate"]; rates.append(p*100); ns.append(int(r["n"]))
        errs.append(np.sqrt(p*(1-p)/r["n"])*100)
    ax.bar(x+dy, rates, w, label=lab, color=col, yerr=errs, capsize=3, edgecolor="white")
    for xi,v,e,n in zip(x+dy, rates, errs, ns):
        ax.text(xi, v+e+1.2, f"{v:.1f}%\n(n={n})", ha="center", va="bottom", fontsize=7.5)
ax.set_xticks(x); ax.set_xticklabels(["DEPTOR low","DEPTOR high"])
ax.set_ylabel("DSS event rate (%)")
ax.set_title("DEPTOR × radiotherapy: event rate reverses\n(qualitative interaction ⇒ predictive biomarker)")
ax.legend(loc="upper left", fontsize=9); ax.set_ylim(0, 58)
plt.tight_layout(); plt.savefig(os.path.join(OUT,"F5_four_cell_event_rate.png")); plt.close()

print("\nSaved to", OUT); print("DONE.")
