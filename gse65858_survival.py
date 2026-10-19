#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GSE65858 (n=270, HNSCC, Illumina) — deconvolution-free survival + mediation.

The NNLS deconvolution degenerated to equal proportions (0.143 each) in TWO
independent attempts (TCGA and GSE65858), i.e. simple NNLS across platforms is
not usable here. So the primary analysis below does NOT depend on it:

  * DEPTOR (probe ILMN_1756685/ILMN_2172755, old symbol DEPDC6) vs OS/PFS
  * stratified by treatment (multi / mono / palliative) and HPV (HPV16 / Other / Negative)
  * mediation: DEPTOR -> hypoxia score -> OS/PFS

Deconvolution-derived fractions are reported but flagged as unusable.
"""
import os, numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
import warnings; warnings.filterwarnings("ignore")

BASE = "WorkBuddy/2026-10-17-11-15-46"
OUT = os.path.join(BASE,"data","results_deconv65858"); os.makedirs(OUT, exist_ok=True)

d = pd.read_csv(os.path.join(OUT,"gse65858_deconv_sample.csv"), index_col=0)
print("samples:", len(d))

# robust survival parsing
def num(s): return pd.to_numeric(pd.Series(s).astype(str).str.strip().replace({"NA":np.nan,"":np.nan,"nan":np.nan}), errors="coerce")
def evt(s):
    x = pd.Series(s).astype(str).str.strip().str.upper()
    return np.where(x.isin(["TRUE","1","1.0","DEAD","DECEASED","YES"]),1,
           np.where(x.isin(["FALSE","0","0.0","ALIVE","LIVING","NO"]),0,np.nan))

d["OS_t"] = num(d["os"]).values
d["OS_e"] = evt(d["os_event"])
d["PFS_t"] = num(d["pfs"]).values
d["PFS_e"] = evt(d["pfs_event"])
print("OS usable:", (d.OS_t.notna() & pd.Series(d.OS_e).notna()).sum(),
      "events:", int(np.nansum(d.OS_e)))
print("PFS usable:", (d.PFS_t.notna() & pd.Series(d.PFS_e).notna()).sum(),
      "events:", int(np.nansum(d.PFS_e)))

d["treatment2"] = d["treatment"].astype(str).str.lower().str.strip()
d["hpv2"] = d["hpv_dna"].astype(str).str.strip()
print("treatment:", d.treatment2.value_counts().to_dict())
print("hpv:", d.hpv2.value_counts().to_dict())

def cox(df, t, e, col, label=""):
    dd = df.dropna(subset=[t,e,col]).copy()
    if len(dd) < 30: return None
    dd["z"] = (dd[col]-dd[col].mean())/dd[col].std()
    cph = CoxPHFitter()
    try:
        cph.fit(dd[[t,e,"z"]], duration_col=t, event_col=e)
        s = cph.summary.loc["z"]
        out = dict(label=label or col, endpoint=t, n=len(dd),
                   HR=round(s["exp(coef)"],3), lo=round(s["exp(coef) lower 95%"],3),
                   hi=round(s["exp(coef) upper 95%"],3), p=round(s["p"],4))
        print(f"  {label:28s} {t:4s} n={len(dd):3d} HR/SD={s['exp(coef)']:.2f} "
              f"({s['exp(coef) lower 95%']:.2f}-{s['exp(coef) upper 95%']:.2f}) p={s['p']:.3g}")
        return out
    except Exception as ex:
        print("  fail", label, t, str(ex)[:50]); return None

print("\n=== whole cohort ===")
rows=[]
for col in ["DEPTOR","hypoxia","NK_sig","purity_marker"]:
    if col not in d.columns: continue
    for t,e in [("OS_t","OS_e"),("PFS_t","PFS_e")]:
        r=cox(d,t,e,col,col); 
        if r: rows.append(r)

print("\n=== by treatment ===")
for tv in ["multi","mono","palliative"]:
    sub=d[d.treatment2==tv]
    if len(sub)<30: continue
    for t,e in [("OS_t","OS_e"),("PFS_t","PFS_e")]:
        r=cox(sub,t,e,"DEPTOR",f"DEPTOR | {tv} (n={len(sub)})")
        if r: rows.append(r)

print("\n=== by HPV ===")
for hv in d.hpv2.dropna().unique():
    sub=d[d.hpv2==hv]
    if len(sub)<30: continue
    for t,e in [("OS_t","OS_e"),("PFS_t","PFS_e")]:
        r=cox(sub,t,e,"DEPTOR",f"DEPTOR | {hv} (n={len(sub)})")
        if r: rows.append(r)

pd.DataFrame(rows).to_csv(os.path.join(OUT,"gse65858_cox_final.csv"), index=False)

# ---------- mediation ----------
print("\n=== mediation: DEPTOR -> hypoxia -> OS/PFS ===")
med=[]
for t,e in [("OS_t","OS_e"),("PFS_t","PFS_e")]:
    dd=d.dropna(subset=[t,e,"DEPTOR","hypoxia"]).copy()
    if len(dd)<80: continue
    dd["X"]=(dd.DEPTOR-dd.DEPTOR.mean())/dd.DEPTOR.std()
    dd["M"]=(dd.hypoxia-dd.hypoxia.mean())/dd.hypoxia.std()
    a=np.polyfit(dd.X,dd.M,1)[0]
    c1=CoxPHFitter(); c1.fit(dd[[t,e,"X"]],duration_col=t,event_col=e)
    lc=np.log(c1.summary.loc["X","exp(coef)"]); pc=c1.summary.loc["X","p"]
    c2=CoxPHFitter(); c2.fit(dd[[t,e,"X","M"]],duration_col=t,event_col=e)
    lc2=np.log(c2.summary.loc["X","exp(coef)"]); pc2=c2.summary.loc["X","p"]
    lb=np.log(c2.summary.loc["M","exp(coef)"]); pb=c2.summary.loc["M","p"]
    prop=(lc-lc2)/lc if lc not in (0,np.nan) else np.nan
    med.append(dict(endpoint=t,n=len(dd),a=round(a,3),total_logHR=round(lc,3),p_total=round(pc,4),
                    direct=round(lc2,3),p_direct=round(pc2,4),b=round(lb,3),p_b=round(pb,4),
                    mediated=round(prop,3) if prop==prop else np.nan))
    print(f"  {t:4s} n={len(dd)} a={a:+.2f} total={lc:+.3f}(p={pc:.3g}) direct={lc2:+.3f}(p={pc2:.3g}) b={lb:+.3f}(p={pb:.3g}) mediated={prop:.1%}")
pd.DataFrame(med).to_csv(os.path.join(OUT,"gse65858_mediation_final.csv"), index=False)

# ---------- KM: DEPTOR high vs low, by treatment ----------
d["DEPTOR_g"] = np.where(d.DEPTOR > d.DEPTOR.median(), "high","low")
fig, axes = plt.subplots(1,3, figsize=(15,4.3))
for ax,tv in zip(axes, ["all","multi","mono"]):
    sub = d if tv=="all" else d[d.treatment2==tv]
    s = sub.dropna(subset=["OS_t","OS_e","DEPTOR_g"])
    if len(s)<20: continue
    kmf=KaplanMeierFitter()
    for g in ["high","low"]:
        m=s.DEPTOR_g==g
        if m.sum()<5: continue
        kmf.fit(s.loc[m,"OS_t"], s.loc[m,"OS_e"], label=f"{g} (n={m.sum()})")
        kmf.plot_survival_function(ax=ax, ci_show=False)
    a=s[s.DEPTOR_g=="high"]; b=s[s.DEPTOR_g=="low"]
    lr=logrank_test(a.OS_t,b.OS_t,a.OS_e,b.OS_e)
    ax.set_title(f"{'all' if tv=='all' else tv} (n={len(s)})\nlog-rank p={lr.p_value:.3g}", fontsize=9)
    ax.set_xlabel("OS (days)"); ax.set_ylabel("survival")
plt.suptitle("GSE65858: DEPTOR high vs low, OS", fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(OUT,"gse65858_KM_DEPTOR_OS.png"), dpi=140); plt.close()

# scatter DEPTOR vs hypoxia
s=d.dropna(subset=["DEPTOR","hypoxia"])
fig,ax=plt.subplots(figsize=(5.5,4.3))
colmap={"HPV16":"#d62728","Other HPV":"#ff7f0e","Negative":"#1f77b4"}
for hv in s.hpv2.dropna().unique():
    m=s.hpv2==hv
    ax.scatter(s.loc[m,"DEPTOR"],s.loc[m,"hypoxia"],s=16,alpha=.7,c=colmap.get(hv,"#999"),label=hv)
rho,p=spearmanr(s.DEPTOR,s.hypoxia)
ax.set_xlabel("DEPTOR (log2)"); ax.set_ylabel("hypoxia score (z)")
ax.set_title(f"GSE65858: DEPTOR vs hypoxia\nrho={rho:+.2f} (p={p:.2g})", fontsize=9)
ax.legend(fontsize=7); plt.tight_layout()
plt.savefig(os.path.join(OUT,"gse65858_DEPTOR_hypoxia.png"),dpi=140); plt.close()

print("\nSaved to", OUT, "DONE.")
