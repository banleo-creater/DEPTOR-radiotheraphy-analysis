#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Purity-adjusted re-analysis of the "DEPTOR -> hypoxia -> radiotherapy benefit"
chain in TCGA-HNSC.

Why this script exists
----------------------
A NNLS deconvolution with only ~57 marker genes was unstable (it swung between
"everything is malignant" and "all 7 types equal"), so it cannot split bulk
DEPTOR into tumour vs immune quantitatively. Instead we use a more robust
approach:

  1. cell-type marker scores (tumour-purity proxy, myeloid, NK, ...)
  2. PARTIAL Spearman correlations controlling for tumour purity
  3. mediation: DEPTOR -> hypoxia -> DSS (Cox-based, log-HR attenuation)
  4. what actually predicts DSS in RT-treated patients

Key question: is the DEPTOR-hypoxia link independent of purity, and is the
DEPTOR-NK link simply a purity artefact?
"""
import os, numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
import warnings; warnings.filterwarnings("ignore")

BASE = "WorkBuddy/2026-10-17-11-15-46"
OUT = os.path.join(BASE, "data", "results_deconv"); os.makedirs(OUT, exist_ok=True)

SETS = {
 "purity": ["KRT5","KRT6A","KRT14","KRT8","KRT18","EPCAM","TP63","KRT19","SFN","KRT17"],
 "myeloid":["LYZ","CD68","CD163","CSF1R","CD14","ITGAM","MRC1","VSIG4","AIF1","C1QA","C1QB","FCGR1A","SPI1","LST1","TYROBP"],
 "Tcell":  ["CD3D","CD3E","CD8A","CD4","IL7R","CD2","CD3G","CD28"],
 "Bcell":  ["CD79A","MS4A1","JCHAIN","MZB1","CD19"],
 "NKs":    ["NKG7","GNLY","KLRD1","PRF1","FCGR3A","NCAM1","KLRF1","XCL1","XCL2","TBX21","EOMES"],
 "hypoxia":["HIF1A","CA9","VEGFA","SLC2A1","LDHA","PGK1","ADM","BNIP3","PDK1"],
 "stroma": ["COL1A1","COL1A2","ACTA2","PDGFRA","DCN","LUM","FAP","PDGFRB"],
 "endo":   ["PECAM1","VWF","CLDN5","CDH5"],
}

clin = pd.read_csv(os.path.join(BASE,"data","tcga_hnsc_pca_deptor.csv"))
dec  = pd.read_csv(os.path.join(BASE,"data","tcga_hnsc_deconv_genes.csv"))
nk   = pd.read_csv(os.path.join(BASE,"data","tcga_hnsc_nkg2d_nk.csv"))
df = clin.merge(dec,on="sampleId",how="inner").merge(nk,on="sampleId",how="inner")
df["RT"]=df.RADIATION_THERAPY.astype(str).str.strip()
df["DEPTOR"]=pd.to_numeric(df.DEPTOR_RSEM,errors="coerce")
df["HPV"]=np.where(df.SUBTYPE.astype(str).str.contains("HPV-"),"HPV-",
          np.where(df.SUBTYPE.astype(str).str.contains("HPV\\+"),"HPV+","NA"))
for t in ["OS","PFS","DSS"]:
    df[f"{t}_event"]=df[f"{t}_STATUS"].astype(str).str.upper().str.strip().str.startswith("1").astype(int)
    df[f"{t}_MONTHS"]=pd.to_numeric(df[f"{t}_MONTHS"],errors="coerce")

def zscore_set(cols):
    zs=[]
    for g in cols:
        if g in df.columns:
            v=pd.to_numeric(df[g],errors="coerce"); df[g]=v
            zs.append((v-v.mean())/v.std())
    return pd.concat(zs,axis=1).mean(axis=1)
for k,v in SETS.items(): df["s_"+k]=zscore_set(v)
df["s_immune"]=df[["s_Tcell","s_Bcell","s_NKs","s_myeloid"]].mean(axis=1)
print("samples:", len(df))

def partial_spearman(x,y,cov):
    d=pd.DataFrame({"x":x,"y":y,"c":cov}).dropna()
    if len(d)<20: return np.nan,np.nan,len(d)
    rx,ry,rc=rankdata(d.x),rankdata(d.y),rankdata(d.c)
    ex=rx-np.polyval(np.polyfit(rc,rx,1),rc)
    ey=ry-np.polyval(np.polyfit(rc,ry,1),rc)
    r,p=spearmanr(ex,ey); return r,p,len(d)

# ---------- 1) crude vs purity-adjusted ----------
print("\n=== crude vs partial (| tumour purity) ===")
rows=[]
for y,name in [("s_hypoxia","Hypoxia"),("s_NKs","NK signature"),("s_myeloid","Myeloid"),
               ("s_immune","Immune"),("s_Tcell","Tcell"),("s_Bcell","Bcell"),
               ("s_stroma","Stroma"),("s_endo","Endo")]:
    s=df[["DEPTOR",y]].dropna()
    r0,p0=spearmanr(s.DEPTOR,s[y])
    r1,p1,n1=partial_spearman(df.DEPTOR,df[y],df.s_purity)
    rows.append(dict(target=name, crude_rho=round(r0,3), crude_p=round(p0,4),
                     adj_rho=round(r1,3), adj_p=round(p1,4), n=n1))
    print(f"  {name:12s} crude={r0:+.3f}(p={p0:.2g}) -> adj={r1:+.3f}(p={p1:.2g})  n={n1}")
pd.DataFrame(rows).to_csv(os.path.join(OUT,"purity_adjusted_correlations.csv"),index=False)

r,p=spearmanr(df[["DEPTOR","s_purity"]].dropna().DEPTOR, df[["DEPTOR","s_purity"]].dropna().s_purity)
print(f"\n  DEPTOR vs tumour purity: rho={r:+.3f} (p={p:.2g})  <- DEPTOR high = LOW purity")

# ---------- 2) RT=Yes partials ----------
print("\n=== RT=Yes ===")
rt=df[df.RT=="Yes"]
for y,name in [("s_hypoxia","Hypoxia"),("s_NKs","NK"),("s_myeloid","Myeloid")]:
    s=rt[["DEPTOR",y]].dropna(); r0,p0=spearmanr(s.DEPTOR,s[y])
    r1,p1,n1=partial_spearman(rt.DEPTOR,rt[y],rt.s_purity)
    print(f"  {name:10s} crude={r0:+.3f}(p={p0:.2g}) -> adj={r1:+.3f}(p={p1:.2g})")

# ---------- 3) what predicts DSS in RT=Yes ----------
print("\n=== DSS predictors in RT=Yes (per SD) ===")
pred=[]
for var in ["s_hypoxia","s_NKs","s_myeloid","s_immune","s_purity","DEPTOR"]:
    d=rt.dropna(subset=["DSS_MONTHS","DSS_event",var]).copy()
    if len(d)<40: continue
    d["z"]=(d[var]-d[var].mean())/d[var].std()
    cph=CoxPHFitter()
    try:
        cph.fit(d[["DSS_MONTHS","DSS_event","z"]],duration_col="DSS_MONTHS",event_col="DSS_event")
        s=cph.summary.loc["z"]
        pred.append(dict(var=var, HR=round(s["exp(coef)"],3), lo=round(s["exp(coef) lower 95%"],3),
                         hi=round(s["exp(coef) upper 95%"],3), p=round(s["p"],4), n=len(d)))
        print(f"  {var:12s} HR/SD={s['exp(coef)']:.2f} ({s['exp(coef) lower 95%']:.2f}-{s['exp(coef) upper 95%']:.2f}) p={s['p']:.3g}")
    except Exception as e: print("  fail",var,str(e)[:50])
pd.DataFrame(pred).to_csv(os.path.join(OUT,"dss_predictors_rt.csv"),index=False)

# ---------- 4) mediation DEPTOR -> hypoxia -> DSS ----------
print("\n=== Mediation: DEPTOR -> Hypoxia -> DSS (RT=Yes) ===")
med=[]
for X in ["DEPTOR"]:
    for t in ["DSS","OS","PFS"]:
        d=rt.dropna(subset=[t+"_MONTHS",t+"_event",X,"s_hypoxia"]).copy()
        if len(d)<80: continue
        d["X"]=(d[X]-d[X].mean())/d[X].std()
        d["M"]=(d.s_hypoxia-d.s_hypoxia.mean())/d.s_hypoxia.std()
        a=np.polyfit(d.X,d.M,1)[0]
        c1=CoxPHFitter(); c1.fit(d[[t+"_MONTHS",t+"_event","X"]],duration_col=t+"_MONTHS",event_col=t+"_event")
        lc=np.log(c1.summary.loc["X","exp(coef)"]); pc=c1.summary.loc["X","p"]
        c2=CoxPHFitter(); c2.fit(d[[t+"_MONTHS",t+"_event","X","M"]],duration_col=t+"_MONTHS",event_col=t+"_event")
        lb=np.log(c2.summary.loc["M","exp(coef)"]); pb=c2.summary.loc["M","p"]
        lc2=np.log(c2.summary.loc["X","exp(coef)"]); pc2=c2.summary.loc["X","p"]
        prop=(lc-lc2)/lc if lc not in (0,np.nan) else np.nan
        med.append(dict(endpoint=t,n=len(d),a_path=round(a,3),logHR_total=round(lc,3),p_total=round(pc,4),
                        logHR_direct=round(lc2,3),p_direct=round(pc2,4),b_logHR=round(lb,3),p_b=round(pb,4),
                        mediated_prop=round(prop,3) if prop==prop else np.nan))
        print(f"  {t:4s} n={len(d)} a={a:+.2f} total={lc:+.3f}(p={pc:.3g}) direct={lc2:+.3f}(p={pc2:.3g}) "
              f"b={lb:+.3f}(p={pb:.3g}) mediated={prop:.1%}" if prop==prop else "")
pd.DataFrame(med).to_csv(os.path.join(OUT,"mediation_purity_aware.csv"),index=False)

# ---------- figures ----------
fig, axes = plt.subplots(1,3, figsize=(14,4.2))
for ax,(y,lab) in zip(axes,[("s_purity","tumour purity"),("s_hypoxia","hypoxia"),("s_NKs","NK signature")]):
    s=df[["DEPTOR",y]].dropna()
    ax.scatter(s.DEPTOR,s[y],s=12,alpha=.5)
    r0,p0=spearmanr(s.DEPTOR,s[y])
    r1,p1,_=partial_spearman(df.DEPTOR,df[y],df.s_purity)
    ax.set_xlabel("DEPTOR (log2 RSEM)"); ax.set_ylabel(lab+" (z)")
    ax.set_title(f"crude rho={r0:+.2f} (p={p0:.1g})\npartial |purity rho={r1:+.2f} (p={p1:.1g})", fontsize=8)
plt.suptitle("TCGA-HNSC: which DEPTOR associations survive tumour-purity adjustment?", fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(OUT,"purity_adjusted_scatter.png"),dpi=140); plt.close()

if rows:
    t=pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7,4))
    y=np.arange(len(t))
    ax.barh(y-0.18,t.crude_rho,0.36,label="crude",color="#999999")
    ax.barh(y+0.18,t.adj_rho,0.36,label="adjusted | purity",color="#1f77b4")
    ax.axvline(0,color="k",lw=.8); ax.set_yticks(y); ax.set_yticklabels(t.target)
    ax.set_xlabel("Spearman rho with DEPTOR"); ax.legend(fontsize=8)
    ax.set_title("Which DEPTOR associations survive purity adjustment?")
    plt.tight_layout(); plt.savefig(os.path.join(OUT,"purity_adjusted_bars.png"),dpi=140); plt.close()

print("\nSaved to", OUT, "DONE.")
