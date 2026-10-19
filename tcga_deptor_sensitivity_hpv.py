#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TCGA-HNSC (PanCan Atlas 2018) DEPTOR — SENSITIVITY + HPV stratification.

(1) Sensitivity analysis of the DEPTOR->radiotherapy-benefit finding:
      - DEPTOR as CONTINUOUS variable (Cox HR per 1 SD), in All / RT-No / RT-Yes
      - DEPTOR dichotomised at the OPTIMAL cutoff (maxstat grid ~ survminer::surv_cutpoint)
        and compared head-to-head with the original MEDIAN split.
(2) HPV stratification (SUBTYPE = HNSC_HPV- / HNSC_HPV+):
      - re-run median-split + continuous + optimal-cutoff within HPV- and HPV+
      - test whether the radiotherapy benefit (DEPTOR x RT interaction) concentrates in HPV-
"""
import os, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

BASE = "WorkBuddy/2026-10-17-11-15-46"
OUT  = os.path.join(BASE, "data", "results_tcga"); os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(os.path.join(BASE, "data", "tcga_hnsc_pca_deptor.csv"))

# ---- clean ----
df["DEPTOR"] = pd.to_numeric(df["DEPTOR_RSEM"], errors="coerce")
def status_to_event(s): return s.astype(str).str.startswith("1").astype(int)
for ep, tcol, scol in [("OS","OS_MONTHS","OS_STATUS"),("PFS","PFS_MONTHS","PFS_STATUS"),("DSS","DSS_MONTHS","DSS_STATUS")]:
    df[f"{ep}_T"] = pd.to_numeric(df[tcol], errors="coerce")
    df[f"{ep}_E"] = status_to_event(df[scol])
df["RT"] = df["RADIATION_THERAPY"].astype(str).str.strip().str.title()
df["RT_yes"] = (df["RT"]=="Yes").astype(int)
df["AGE"] = pd.to_numeric(df["AGE"], errors="coerce")
df["SEX_M"] = (df["SEX"].astype(str).str.upper().str.startswith("M")).astype(int)
df["STAGE_IV"] = df["AJCC_PATHOLOGIC_TUMOR_STAGE"].map(lambda s: 1 if "IV" in str(s).upper() else 0)
# HPV from SUBTYPE
df["HPV"] = df["SUBTYPE"].map(lambda s: "HPV+" if str(s).endswith("HPV+") else ("HPV-" if str(s).endswith("HPV-") else np.nan))

med = df["DEPTOR"].median()
df["DEPTOR_high"] = (df["DEPTOR"] >= med).astype(int)
print(f"N DEPTOR non-null: {df['DEPTOR'].notna().sum()} | median RSEM={med:.2f}")
print("RT:", df["RT"].value_counts(dropna=False).to_dict())
print("HPV:", df["HPV"].value_counts(dropna=False).to_dict())

endpoints = ["OS","PFS","DSS"]

# ---------- helper: univariate Cox (single covariate) ----------
def cox_uni(d, ep, cov):
    d = d.dropna(subset=[f"{ep}_T", f"{ep}_E", cov]).copy()
    d = d[d[f"{ep}_T"]>0]
    if d[f"{ep}_E"].sum() < 3 or d[cov].nunique() < 2: return None
    dd = d[[f"{ep}_T", f"{ep}_E", cov]].rename(columns={f"{ep}_T":"T", f"{ep}_E":"E"})
    cph = CoxPHFitter(); cph.fit(dd, duration_col="T", event_col="E")
    hr = float(np.exp(cph.params_[cov])); p = float(cph.summary.loc[cov,"p"])
    lo, hi = [float(x) for x in np.exp(cph.confidence_intervals_.loc[cov]).values]
    return dict(n=len(dd), events=int(d[f"{ep}_E"].sum()), HR=hr, lo=lo, hi=hi, p=p)

# ---------- helper: optimal cutoff (maxstat grid) ----------
def optimal_cutpoint(t, e, x, minprop=0.10, maxprop=0.90):
    x = np.asarray(x, float); order = np.argsort(x)
    xs, ts, es = x[order], np.asarray(t)[order], np.asarray(e)[order]
    n = len(x); lo = int(np.ceil(minprop*n)); hi = int(np.floor(maxprop*n))
    best = None
    for k in range(lo, hi+1):
        lr = logrank_test(ts[:k], ts[k:], es[:k], es[k:])
        s = lr.test_statistic
        if best is None or s > best[0]:
            best = (s, float(xs[k-1]), k, float(lr.p_value))
    return best  # (chi2, cutoff, k, logrank_p)

# ============================================================
# PART 1 — SENSITIVITY (whole cohort)
# ============================================================
print("\n" + "="*72)
print("PART 1a — DEPTOR CONTINUOUS (Cox HR per 1 SD)")
print("="*72)
rows = []
for ep in endpoints:
    for strat, sub in [("All", df), ("RT No", df[df.RT=="No"]), ("RT Yes", df[df.RT=="Yes"])]:
        d = sub.dropna(subset=[f"{ep}_T", f"{ep}_E", "DEPTOR"]).copy()
        if len(d) < 20: continue
        x = d["DEPTOR"].values.astype(float)
        sd = x.std(); z = (x - x.mean())/sd
        dd = d[[f"{ep}_T", f"{ep}_E"]].copy(); dd["DEPTOR_z"] = z
        cph = CoxPHFitter(); cph.fit(dd, duration_col=f"{ep}_T", event_col=f"{ep}_E")
        hr = float(np.exp(cph.params_["DEPTOR_z"])); p = float(cph.summary.loc["DEPTOR_z","p"])
        lo, hi = [float(v) for v in np.exp(cph.confidence_intervals_.loc["DEPTOR_z"]).values]
        # also raw per-unit
        dd2 = d[[f"{ep}_T", f"{ep}_E"]].copy(); dd2["DEPTOR"] = x
        cph2 = CoxPHFitter(); cph2.fit(dd2, duration_col=f"{ep}_T", event_col=f"{ep}_E")
        hr_unit = float(np.exp(cph2.params_["DEPTOR"]))
        rows.append(dict(endpoint=ep, stratum=strat, n=len(d), DEPTOR_SD=round(sd,1),
                        HR_per_SD=round(hr,3), lo_per_SD=round(lo,3), hi_per_SD=round(hi,3),
                        p_per_SD=round(p,4), HR_per_unit=hr_unit))
        print(f"  {ep:3s} | {strat:6s}: n={len(d)}  HR/SD={hr:.3f} ({lo:.3f}-{hi:.3f}) p={p:.3g}  | HR/unit={hr_unit:.4g}")
pd.DataFrame(rows).to_csv(os.path.join(OUT,"sens_continuous.csv"), index=False)

print("\n" + "="*72)
print("PART 1b — OPTIMAL cutoff (maxstat) vs MEDIAN split")
print("="*72)
rows = []
km_best = {}
for ep in endpoints:
    for strat, sub in [("All", df), ("RT No", df[df.RT=="No"]), ("RT Yes", df[df.RT=="Yes"])]:
        d = sub.dropna(subset=[f"{ep}_T", f"{ep}_E", "DEPTOR"]).copy()
        d = d[d[f"{ep}_T"]>0]
        if len(d) < 20: continue
        t, e, x = d[f"{ep}_T"].values, d[f"{ep}_E"].values, d["DEPTOR"].values
        # median split
        m = df["DEPTOR"].median()
        d["grp_med"] = (x >= m).astype(int)
        lr_med = logrank_test(t[d.grp_med==1], t[d.grp_med==0], e[d.grp_med==1], e[d.grp_med==0])
        cx_med = cox_uni(d.assign(DEPTOR_high=d.grp_med), ep, "DEPTOR_high")
        # optimal cutoff
        chi2, cut, k, plr = optimal_cutpoint(t, e, x)
        d["grp_opt"] = (x > cut).astype(int)
        lr_opt = logrank_test(t[d.grp_opt==1], t[d.grp_opt==0], e[d.grp_opt==1], e[d.grp_opt==0])
        cx_opt = cox_uni(d.assign(DEPTOR_high=d.grp_opt), ep, "DEPTOR_high")
        rows.append(dict(endpoint=ep, stratum=strat, n=len(d),
            median_RSEM=round(m,1), med_HR=round(cx_med["HR"],3), med_p=round(cx_med["p"],4), med_logrank_p=round(float(lr_med.p_value),4),
            opt_cutoff_RSEM=round(cut,1), opt_fraction_high=round(k/len(d),3), opt_HR=round(cx_opt["HR"],3),
            opt_p=round(cx_opt["p"],4), opt_logrank_p=round(plr,4)))
        km_best[(ep,strat)] = (t, e, d["grp_opt"].values, cut)
        print(f"  {ep:3s} | {strat:6s}: n={len(d)}")
        print(f"        MEDIAN  cut={m:.1f}  HR={cx_med['HR']:.2f} (p={cx_med['p']:.3g})  logrank p={float(lr_med.p_value):.3g}")
        print(f"        OPTIMAL cut={cut:.1f} (hi frac {k/len(d):.2f}) HR={cx_opt['HR']:.2f} (p={cx_opt['p']:.3g})  logrank p={plr:.3g}")
pd.DataFrame(rows).to_csv(os.path.join(OUT,"sens_optimal_vs_median.csv"), index=False)

# ============================================================
# PART 2 — HPV STRATIFICATION
# ============================================================
print("\n" + "="*72)
print("PART 2 — HPV stratification (SUBTYPE)")
print("="*72)
rows = []
for hpv, subg in [("HPV-", df[df.HPV=="HPV-"]), ("HPV+", df[df.HPV=="HPV+"])]:
    print(f"\n--- {hpv} (n={len(subg)}, RT Yes={int((subg.RT=='Yes').sum())}, RT No={int((subg.RT=='No').sum())}) ---")
    for ep in endpoints:
        # median split within this HPV group
        d = subg.dropna(subset=[f"{ep}_T", f"{ep}_E", "DEPTOR"]).copy()
        if len(d) < 10: continue
        mm = d["DEPTOR"].median()
        d["DH"] = (d["DEPTOR"] >= mm).astype(int)
        for strat, ss in [("All", d), ("RT No", d[d.RT=="No"]), ("RT Yes", d[d.RT=="Yes"])]:
            r = cox_uni(ss, ep, "DH")
            if r is None: continue
            rows.append(dict(HPV=hpv, endpoint=ep, stratum=strat, n=r["n"],
                             med_RSEM=round(mm,1), HR=round(r["HR"],3), lo=round(r["lo"],3),
                             hi=round(r["hi"],3), p=round(r["p"],4)))
            print(f"    {ep:3s} | {strat:6s}: n={r['n']} HR(hi vs lo)={r['HR']:.2f} ({r['lo']:.2f}-{r['hi']:.2f}) p={r['p']:.3g}")
        # continuous per SD within whole HPV group
        x = d["DEPTOR"].values.astype(float); z=(x-x.mean())/x.std()
        dd=d[[f"{ep}_T",f"{ep}_E"]].copy(); dd["z"]=z
        cph=CoxPHFitter(); cph.fit(dd,duration_col=f"{ep}_T",event_col=f"{ep}_E")
        hr=float(np.exp(cph.params_["z"])); p=float(cph.summary.loc["z","p"])
        rows.append(dict(HPV=hpv, endpoint=ep, stratum="All(cont/SD)", n=len(d),
                         med_RSEM=round(mm,1), HR=round(hr,3), lo=np.nan, hi=np.nan, p=round(p,4)))
        print(f"    {ep:3s} | All(cont/SD): HR/SD={hr:.3f} p={p:.3g}")
pd.DataFrame(rows).to_csv(os.path.join(OUT,"hpv_stratified.csv"), index=False)

print("\n" + "-"*72)
print("PART 2b — Does RT benefit concentrate in HPV-? (DEPTOR x RT interaction)")
print("-"*72)
int_rows=[]
for hpv, subg in [("HPV-", df[df.HPV=="HPV-"]), ("HPV+", df[df.HPV=="HPV+"])]:
    for ep in endpoints:
        d = subg.dropna(subset=[f"{ep}_T", f"{ep}_E", "RT_yes", "DEPTOR"]).copy()
        d = d[d[f"{ep}_T"]>0]
        d["DH"] = (d["DEPTOR"] >= d["DEPTOR"].median()).astype(int)
        d["INT"] = d["DH"]*d["RT_yes"]
        if d["INT"].sum() < 3: continue
        dd = d[[f"{ep}_T",f"{ep}_E","DH","RT_yes","INT"]].rename(columns={f"{ep}_T":"T",f"{ep}_E":"E"})
        cph=CoxPHFitter(); cph.fit(dd, duration_col="T", event_col="E")
        p_int=float(cph.summary.loc["INT","p"]); hr_int=float(np.exp(cph.params_["INT"]))
        # continuous interaction too
        x=d["DEPTOR"].values.astype(float); z=(x-x.mean())/x.std()
        dd2=d[[f"{ep}_T",f"{ep}_E","RT_yes"]].copy(); dd2["z"]=z; dd2["INTz"]=z*d["RT_yes"].values
        cph2=CoxPHFitter(); cph2.fit(dd2.rename(columns={f"{ep}_T":"T",f"{ep}_E":"E"}), duration_col="T", event_col="E")
        p_intz=float(cph2.summary.loc["INTz","p"]); hr_intz=float(np.exp(cph2.params_["INTz"]))
        int_rows.append(dict(HPV=hpv, endpoint=ep, n=len(d),
                             dicho_INT_HR=round(hr_int,3), dicho_INT_p=round(p_int,4),
                             cont_INT_HRperSD=round(hr_intz,3), cont_INT_p=round(p_intz,4)))
        print(f"  {hpv} {ep:3s}: n={len(d)}  dicho INT(DHxRT) HR={hr_int:.3f} p={p_int:.3g}  | cont INT(zxRT) HR/SD={hr_intz:.3f} p={p_intz:.3g}")
pd.DataFrame(int_rows).to_csv(os.path.join(OUT,"hpv_interaction.csv"), index=False)

# ---- Figures ----
# Forest: optimal vs median HR by endpoint x stratum (whole cohort)
sens = pd.read_csv(os.path.join(OUT,"sens_optimal_vs_median.csv"))
fig, ax = plt.subplots(figsize=(8,6))
labels=[]; ys=[]; y=0
for ep in endpoints:
    for sname in ["All","RT No","RT Yes"]:
        r = sens[(sens.endpoint==ep)&(sens.stratum==sname)]
        if r.empty: continue
        r=r.iloc[0]
        for val, lab, col in [(r.med_HR, "median", "#E24A33"), (r.opt_HR, "optimal", "#348ABD")]:
            lo = max(val*0.5, 0.05); hi = val*1.8
            ax.errorbar(val, y, xerr=[[val-lo],[hi-val]], fmt="o", color=col, capsize=3, ms=5)
            labels.append(f"{ep} | {sname} [{lab}]"); ys.append(y); y+=1
        y+=0.5
ax.axvline(1.0, color="red", ls="--", lw=1)
ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=7)
ax.set_xscale("log"); ax.set_xlabel("HR (DEPTOR high vs low), log scale")
ax.set_title("DEPTOR HR: median split vs optimal cutoff\n(whole cohort; red=median, blue=optimal)")
ax.invert_yaxis(); plt.tight_layout()
plt.savefig(os.path.join(OUT,"sens_forest_median_vs_optimal.png"), dpi=170); plt.close()

# Forest: HPV stratified HR (median split) for PFS/DSS
hpv = pd.read_csv(os.path.join(OUT,"hpv_stratified.csv"))
fig, ax = plt.subplots(figsize=(8,6.5))
labels=[]; ys=[]; y=0
for hpv_grp in ["HPV-","HPV+"]:
    for ep in ["PFS","DSS"]:
        for sname in ["All","RT No","RT Yes"]:
            r = hpv[(hpv.HPV==hpv_grp)&(hpv.endpoint==ep)&(hpv.stratum==sname)]
            if r.empty: continue
            r=r.iloc[0]
            col = "#E24A33" if hpv_grp=="HPV-" else "#2ca02c"
            lo=max(r.HR*0.5,0.05); hi=r.HR*1.8
            ax.errorbar(r.HR, y, xerr=[[r.HR-lo],[hi-r.HR]], fmt="o", color=col, capsize=3, ms=5)
            labels.append(f"{hpv_grp} {ep} | {sname} (n={int(r.n)}, p={r.p:.2g})"); ys.append(y); y+=1
        y+=0.5
ax.axvline(1.0, color="red", ls="--", lw=1)
ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=7)
ax.set_xscale("log"); ax.set_xlabel("HR (DEPTOR high vs low), log scale")
ax.set_title("DEPTOR HR by HPV status (median split)\nred=HPV-, green=HPV+")
ax.invert_yaxis(); plt.tight_layout()
plt.savefig(os.path.join(OUT,"hpv_forest.png"), dpi=170); plt.close()

# KM: RT=Yes, PFS & DSS, by HPV, DEPTOR high/low
from lifelines import KaplanMeierFitter
fig, axes = plt.subplots(2,2, figsize=(12,10))
for i,(ep,hgrp) in enumerate([("PFS","HPV-"),("DSS","HPV-"),("PFS","HPV+"),("DSS","HPV+")]):
    ax=axes[i//2, i%2]
    d = df[(df.HPV==hgrp)&(df.RT=="Yes")].dropna(subset=[f"{ep}_T",f"{ep}_E","DEPTOR"]).copy()
    d["DH"]=(d["DEPTOR"]>=d["DEPTOR"].median()).astype(int)
    for gv,lab in [(0,"DEPTOR low"),(1,"DEPTOR high")]:
        m=d[d.DH==gv]
        if len(m)<3: continue
        kmf=KaplanMeierFitter()
        kmf.fit(m[f"{ep}_T"]/12.0, m[f"{ep}_E"], label=f"{lab} (n={len(m)})")
        kmf.plot_survival_function(ax=ax, ci_show=False, lw=1.8,
                                    color="#E24A33" if gv==1 else "#348ABD")
    ax.set_title(f"{hgrp} RT=Yes — {ep}"); ax.set_xlabel("Years"); ax.set_ylabel("Survival"); ax.grid(alpha=.25)
plt.suptitle("DEPTOR high vs low in RT=Yes, stratified by HPV", y=1.0)
plt.tight_layout(); plt.savefig(os.path.join(OUT,"hpv_KM_RTyes.png"), dpi=170); plt.close()

print("\nSaved to", OUT); print("DONE.")
