#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TCGA-HNSC (PanCancer Atlas 2018) DEPTOR survival analysis stratified by radiotherapy.
Reproduces the OS / PFS(≈PFI) / DSS x (All, RT-No, RT-Yes) framework.
"""
import os, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "WorkBuddy/2026-10-17-11-15-46"
OUT  = os.path.join(BASE,"data","results_tcga"); os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(os.path.join(BASE,"data","tcga_hnsc_pca_deptor.csv"))

# merge GRADE (sample-level)
grade = {r["sampleId"]: r["value"] for r in json.load(open(os.path.join(BASE,"data","pca_grade_s.json")))}
df["GRADE"] = df["sampleId"].map(grade)

# ---- clean ----
df["DEPTOR"] = pd.to_numeric(df["DEPTOR_RSEM"], errors="coerce")
def status_to_event(s):
    return s.astype(str).str.startswith("1").astype(int)
for ep, tcol, scol in [("OS","OS_MONTHS","OS_STATUS"),("PFS","PFS_MONTHS","PFS_STATUS"),("DSS","DSS_MONTHS","DSS_STATUS")]:
    df[f"{ep}_T"] = pd.to_numeric(df[tcol], errors="coerce")
    df[f"{ep}_E"] = status_to_event(df[scol])

df["RT"] = df["RADIATION_THERAPY"].astype(str).str.strip().str.title()
df["AGE"] = pd.to_numeric(df["AGE"], errors="coerce")
df["SEX_M"] = (df["SEX"].astype(str).str.upper().str.startswith("M")).astype(int)

def stage_iv(s):
    s = str(s)
    return 1 if "IV" in s.upper() else 0
df["STAGE_IV"] = df["AJCC_PATHOLOGIC_TUMOR_STAGE"].map(stage_iv)
def grade_high(g):
    g = str(g).upper()
    return 1 if g in ("G3","G4") else 0
df["GRADE_HIGH"] = df["GRADE"].map(grade_high)

# median split (global, among patients with DEPTOR)
med = df["DEPTOR"].median()
df["DEPTOR_high"] = (df["DEPTOR"] >= med).astype(int)
print(f"N with DEPTOR: {df['DEPTOR'].notna().sum()} | median RSEM={med:.1f}")
print("RT distribution:", df["RT"].value_counts().to_dict())
print("DEPTOR high/low:", df["DEPTOR_high"].value_counts().to_dict())

# ---- survival helpers ----
def km_cox(sub, ep, strat_label):
    """Return dict of stats + fitted objects for KM plot."""
    d = sub.dropna(subset=[f"{ep}_T", f"{ep}_E"]).copy()
    d = d[d[f"{ep}_T"] > 0]
    n_hi = int((d["DEPTOR_high"]==1).sum()); n_lo = int((d["DEPTOR_high"]==0).sum())
    if n_hi < 5 or n_lo < 5:
        return None
    t = d[f"{ep}_T"].values; e = d[f"{ep}_E"].values; g = d["DEPTOR_high"].values
    lr = logrank_test(t[g==1], t[g==0], e[g==1], e[g==0])
    p_logrank = lr.p_value
    # Cox univariate
    cx = d[[f"{ep}_T", f"{ep}_E", "DEPTOR_high"]].rename(columns={f"{ep}_T":"T", f"{ep}_E":"E"})
    cph = CoxPHFitter()
    cph.fit(cx, duration_col="T", event_col="E")
    hr = float(np.exp(cph.params_["DEPTOR_high"]))
    ci_lo, ci_hi = [float(x) for x in np.exp(cph.confidence_intervals_.loc["DEPTOR_high"]).values]
    p_cox = float(cph.summary.loc["DEPTOR_high","p"])
    return dict(endpoint=ep, stratum=strat_label, n=n_hi+n_lo, n_low=n_lo, n_high=n_hi,
                events=int(e.sum()), HR=hr, HR_lo=ci_lo, HR_hi=ci_hi,
                p_cox=p_cox, p_logrank=float(p_logrank),
                t=t, e=e, g=g, med_hi=np.median(t[g==1]), med_lo=np.median(t[g==0]))

# ---- main loop ----
strata = {"All": df[df["DEPTOR"].notna()],
          "RT No": df[(df["RT"]=="No") & df["DEPTOR"].notna()],
          "RT Yes": df[(df["RT"]=="Yes") & df["DEPTOR"].notna()]}
endpoints = ["OS","PFS","DSS"]

results = []
km_store = {}
for ep in endpoints:
    for sname, sub in strata.items():
        r = km_cox(sub, ep, sname)
        if r is None:
            print(f"  {ep} | {sname}: insufficient"); continue
        results.append({k:v for k,v in r.items() if k not in ("t","e","g","med_hi","med_lo")})
        km_store[(ep,sname)] = r
        print(f"  {ep:3s} | {sname:6s}: n={r['n']} (lo {r['n_low']}/hi {r['n_high']}) "
              f"events={r['events']}  HR(high vs low)={r['HR']:.2f} "
              f"({r['HR_lo']:.2f}-{r['HR_hi']:.2f})  Cox p={r['p_cox']:.3g}  logrank p={r['p_logrank']:.3g}")

res_df = pd.DataFrame(results)
res_df.to_csv(os.path.join(OUT,"deptor_survival_summary.csv"), index=False)

# ---- interaction test: DEPTOR_high * RT (whole cohort) ----
print("\n=== Interaction (DEPTOR_high x RT), whole cohort ===")
inter_rows=[]
for ep in endpoints:
    d = df.dropna(subset=[f"{ep}_T", f"{ep}_E","RT"]).copy()
    d = d[(d[f"{ep}_T"]>0) & (d["RT"].isin(["Yes","No"])) & d["DEPTOR"].notna()]
    d["RT_yes"] = (d["RT"]=="Yes").astype(int)
    d["INT"] = d["DEPTOR_high"]*d["RT_yes"]
    cx = d[[f"{ep}_T",f"{ep}_E","DEPTOR_high","RT_yes","INT"]].rename(columns={f"{ep}_T":"T",f"{ep}_E":"E"})
    cph = CoxPHFitter(); cph.fit(cx, duration_col="T", event_col="E")
    p_int = float(cph.summary.loc["INT","p"])
    hr_int = float(np.exp(cph.params_["INT"]))
    inter_rows.append(dict(endpoint=ep, n=len(d), HR_interaction=hr_int, p_interaction=p_int))
    print(f"  {ep}: n={len(d)}  HR_int={hr_int:.3f}  p_int={p_int:.3g}")
pd.DataFrame(inter_rows).to_csv(os.path.join(OUT,"deptor_rt_interaction.csv"), index=False)

# ---- multivariate Cox within RT=Yes ----
print("\n=== Multivariate Cox in RT=Yes (DEPTOR adjusted for age/sex/stage/grade) ===")
mv_rows=[]
for ep in endpoints:
    d = df[(df["RT"]=="Yes")].dropna(subset=[f"{ep}_T",f"{ep}_E"]).copy()
    d = d[d[f"{ep}_T"]>0]
    cols = [f"{ep}_T", f"{ep}_E", "DEPTOR_high", "AGE", "SEX_M", "STAGE_IV", "GRADE_HIGH"]
    dd = d[[c for c in cols if c in d.columns]].rename(columns={f"{ep}_T":"T",f"{ep}_E":"E"}).dropna()
    if len(dd) < 20: 
        print(f"  {ep}: insufficient n={len(dd)}"); continue
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(dd, duration_col="T", event_col="E")
    hr = float(np.exp(cph.params_["DEPTOR_high"])); p = float(cph.summary.loc["DEPTOR_high","p"])
    lo,hi = [float(x) for x in np.exp(cph.confidence_intervals_.loc["DEPTOR_high"]).values]
    mv_rows.append(dict(endpoint=ep, n=len(dd), adj_HR=hr, adj_lo=lo, adj_hi=hi, p=p))
    print(f"  {ep}: n={len(dd)}  adj.HR={hr:.2f} ({lo:.2f}-{hi:.2f})  p={p:.3g}")
pd.DataFrame(mv_rows).to_csv(os.path.join(OUT,"deptor_rt_yes_multivariable.csv"), index=False)

# ---- FIGURE: KM grid 3 endpoints x 3 strata ----
colors = {1:"#E24A33", 0:"#348ABD"}
fig, axes = plt.subplots(3, 3, figsize=(15, 13))
row_labels = {"OS":"Overall survival","PFS":"Progression-free survival (≈PFI)","DSS":"Disease-specific survival"}
col_labels = {"All":"All patients","RT No":"Radiation therapy: No","RT Yes":"Radiation therapy: Yes"}
for i, ep in enumerate(endpoints):
    for j, sname in enumerate(["All","RT No","RT Yes"]):
        ax = axes[i,j]; r = km_store.get((ep,sname))
        if r is None:
            ax.axis("off"); continue
        t,e,g = r["t"],r["e"],r["g"]
        for gv,lab in [(0,"DEPTOR low"),(1,"DEPTOR high")]:
            m = g==gv
            kmf = KaplanMeierFitter()
            kmf.fit(t[m]/12.0, e[m], label=f"{lab} (n={m.sum()})")
            kmf.plot_survival_function(ax=ax, ci_show=False, color=colors[gv], lw=1.8)
        ax.set_title(f"{row_labels[ep]}\n{col_labels[sname]}", fontsize=10)
        ax.set_xlabel("Time (years)"); ax.set_ylabel("Survival probability")
        ax.set_ylim(0,1.02); ax.grid(alpha=0.25)
        txt = (f"HR(high vs low)={r['HR']:.2f} ({r['HR_lo']:.2f}–{r['HR_hi']:.2f})\n"
               f"log-rank P={r['p_logrank']:.3g}")
        ax.text(0.03,0.06, txt, transform=ax.transAxes, fontsize=8.5,
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
        ax.legend(fontsize=7, loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(OUT,"deptor_KM_grid_OS_PFS_DSS.png"), dpi=170); plt.close()

# ---- FIGURE: forest plot of HRs ----
fig, ax = plt.subplots(figsize=(7,5))
labels=[]; ys=[]
y=0
for ep in endpoints:
    for sname in ["All","RT No","RT Yes"]:
        r = [x for x in results if x["endpoint"]==ep and x["stratum"]==sname]
        if not r: continue
        r=r[0]
        lo,hi = max(r["HR_lo"],0.1), min(r["HR_hi"],10)
        ax.errorbar(r["HR"], y, xerr=[[r["HR"]-lo],[hi-r["HR"]]], fmt="o", color="#333", capsize=3)
        labels.append(f"{ep} | {sname}  (n={r['n']}, P={r['p_logrank']:.2g})")
        ys.append(y); y+=1
    y+=0.6
ax.axvline(1.0, color="red", ls="--", lw=1)
ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=8)
ax.set_xscale("log"); ax.set_xlabel("Hazard ratio (DEPTOR high vs low), log scale")
ax.set_title("DEPTOR high vs low: HR by endpoint and radiotherapy stratum")
ax.invert_yaxis()
plt.tight_layout(); plt.savefig(os.path.join(OUT,"deptor_HR_forest.png"), dpi=170); plt.close()

# ---- FIGURE: DEPTOR by GRADE (panel B) ----
sub = df.dropna(subset=["DEPTOR","GRADE"])
sub = sub[sub["GRADE"].isin(["G1","G2","G3","G4"])]
plt.figure(figsize=(6,4.5))
groups = ["G1","G2","G3","G4"]
data = [sub[sub["GRADE"]==g]["DEPTOR"].values for g in groups]
plt.boxplot(data, tick_labels=groups, showfliers=False)
for k,g in enumerate(groups):
    x = np.random.normal(k+1, 0.05, len(data[k]))
    plt.scatter(x, data[k], s=6, alpha=0.35, color="#555")
from scipy.stats import kruskal
try:
    H,p = kruskal(*[d for d in data if len(d)>0])
    plt.title(f"DEPTOR expression by histologic grade\nKruskal-Wallis P={p:.3g}")
except Exception: pass
plt.ylabel("DEPTOR (RSEM)"); plt.tight_layout()
plt.savefig(os.path.join(OUT,"deptor_by_grade.png"), dpi=170); plt.close()

# ---- FIGURE: DEPTOR by LVI (panel C) using legacy ANGIOLYMPHATIC_INVASION ----
try:
    leg = pd.read_csv(os.path.join(BASE,"data","tcga_hnsc_deptor.csv"))
    lvi = leg.set_index("patientId")["ANGIOLYMPHATIC_INVASION"].to_dict()
    d2 = df.copy(); d2["LVI"] = d2["patientId"].map(lvi)
    d2 = d2[d2["LVI"].isin(["YES","NO"])]
    if len(d2)>10:
        plt.figure(figsize=(5.5,4.5))
        a = d2[d2["LVI"]=="NO"]["DEPTOR"].dropna(); b = d2[d2["LVI"]=="YES"]["DEPTOR"].dropna()
        plt.boxplot([a,b], tick_labels=[f"No (n={len(a)})",f"Yes (n={len(b)})"], showfliers=False)
        from scipy.stats import mannwhitneyu
        U,pw = mannwhitneyu(a,b)
        plt.title(f"DEPTOR expression by lymphovascular invasion\nMann-Whitney P={pw:.3g}")
        plt.ylabel("DEPTOR (RSEM)"); plt.tight_layout()
        plt.savefig(os.path.join(OUT,"deptor_by_LVI.png"), dpi=170); plt.close()
        print(f"\nLVI panel: NO n={len(a)}, YES n={len(b)}, MWU p={pw:.3g}")
except Exception as e:
    print("LVI panel skipped:", e)

print("\nSaved to", OUT)
print("DONE.")
