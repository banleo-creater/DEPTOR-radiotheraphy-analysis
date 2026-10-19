#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Independent-cohort deconvolution in GSE65858 (n=270 HNSCC, Illumina HT-12 V4,
OS/PFS + HPV + treatment), using the FULL-TRANSCRIPTOME GSE164690 reference and
1,367 informative genes (vs the 57 that made NNLS unstable).

Steps
 1. probe -> gene symbol from GPL10558 platform table
 2. parse GSE65858 series-matrix expression (log2 microarray)
 3. collapse probes to genes (max expression)
 4. NNLS deconvolution with the GSE164690 centroid (7 cell types)
 5. split bulk DEPTOR into tumour vs myeloid/immune contributions
 6. survival + treatment(multi vs mono) interaction
 7. mediation: DEPTOR -> hypoxia -> survival
"""
import os, re, gzip, numpy as np, pandas as pd
from scipy.optimize import nnls
from scipy.stats import spearmanr, rankdata
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
import warnings; warnings.filterwarnings("ignore")

BASE = "WorkBuddy/2026-10-17-11-15-46"
GEO = "<PATH>"
OUT = os.path.join(BASE, "data", "results_deconv65858"); os.makedirs(OUT, exist_ok=True)

# ---------- 1. platform annotation ----------
def load_gpl_symbols(soft_gz):
    sym_col = None; id_col = 0; rows = {}
    with gzip.open(soft_gz, "rt", errors="ignore") as f:
        intab = False
        for line in f:
            if line.startswith("!platform_table_begin"):
                intab = True; continue
            if line.startswith("!platform_table_end"):
                break
            if not intab: continue
            p = line.rstrip("\n").split("\t")
            if sym_col is None:
                # header
                cols = [c.strip().lower() for c in p]
                for cand in ("symbol","ilmn_gene","gene symbol","gene_symbol"):
                    if cand in cols:
                        sym_col = cols.index(cand); break
                if sym_col is None: sym_col = 12
                continue
            if len(p) <= sym_col: continue
            pid = p[id_col].strip(); sym = p[sym_col].strip().strip('"')
            if pid and sym and sym not in ("", "---"):
                rows[pid] = sym.split("///")[0].strip()
    return rows

print("loading GPL10558 annotation ...")
p2s = load_gpl_symbols(os.path.join(BASE,"data","plat","GPL10558.soft.gz"))
print(f"  probe->symbol: {len(p2s)}")

# ---------- 2. expression ----------
def parse_series_matrix(sm_gz):
    with gzip.open(sm_gz, "rt", errors="ignore") as f:
        lines = f.readlines()
    # sample ids
    sids = None
    for line in lines:
        if line.startswith("!Sample_geo_accession"):
            sids = [v.strip().strip('"') for v in line.rstrip("\n").split("\t")[1:]]
            break
    # table
    start = next(i for i,l in enumerate(lines) if l.startswith("!series_matrix_table_begin"))
    end   = next(i for i,l in enumerate(lines) if l.startswith("!series_matrix_table_end"))
    hdr = lines[start+1].rstrip("\n").split("\t")
    ids = [h.strip().strip('"') for h in hdr[1:]]
    data = []
    for l in lines[start+2:end]:
        p = l.rstrip("\n").split("\t")
        data.append([p[0].strip().strip('"')] + [float(x) if x.strip() not in ("","NA","null") else np.nan for x in p[1:]])
    df = pd.DataFrame(data, columns=["ID"]+ids).set_index("ID")
    return df, sids

print("parsing GSE65858 expression ...")
expr, sids = parse_series_matrix(os.path.join(GEO,"GSE65858","GSE65858_series_matrix.txt.gz"))
print(f"  probes x samples: {expr.shape}")

# collapse probes -> gene (max)
expr["symbol"] = [p2s.get(i, None) for i in expr.index]
expr = expr.dropna(subset=["symbol"])
expr["symbol"] = expr["symbol"].replace({"DEPDC6": "DEPTOR"})   # DEPTOR's old symbol
expr = expr.groupby("symbol").max()
print(f"  gene-level: {expr.shape}")

# ---------- 3. clinical ----------
def parse_characteristics(sm_gz):
    with gzip.open(sm_gz, "rt", errors="ignore") as f:
        lines = f.readlines()
    sids = None; d = {}
    for line in lines:
        if line.startswith("!Sample_geo_accession"):
            sids = [v.strip().strip('"') for v in line.rstrip("\n").split("\t")[1:]]
        elif line.startswith("!Sample_characteristics_ch1"):
            vals = [v.strip().strip('"') for v in line.rstrip("\n").split("\t")[1:]]
            if not vals or ":" not in vals[0]: continue
            key = vals[0].split(":")[0].strip()
            d[key] = [v.split(":",1)[-1].strip() for v in vals]
    return pd.DataFrame(d, index=sids) if sids else pd.DataFrame()

clin = parse_characteristics(os.path.join(GEO,"GSE65858","GSE65858_series_matrix.txt.gz"))
print("  clinical keys:", [k for k in clin.columns][:20])

# ---------- 4. deconvolution ----------
ref = pd.read_csv(os.path.join(BASE,"data","gse164690_centroid_full.csv"), index_col=0)
hvg = [l.strip() for l in open(os.path.join(BASE,"data","gse164690_hvg.txt")) if l.strip()]
genes = [g for g in hvg if g in ref.columns and g in expr.index]
print(f"\ngenes shared (ref & GSE65858): {len(genes)}")
Hraw = ref[genes].values.astype(float)
H = Hraw / np.maximum(Hraw.mean(axis=0, keepdims=True), 1e-9)
Y = expr.loc[genes].T.values.astype(float)          # samples x genes, log2 already
Y = np.nan_to_num(Y)
# microarray log2 ~ 2-15; centroid log1p ~ 0-5 -> both log-ish; scale each to mean 1
Yscaled = Y / np.maximum(Y.mean(axis=0, keepdims=True), 1e-9)

props = np.zeros((Y.shape[0], H.shape[0]))
for i in range(Y.shape[0]):
    w,_ = nnls(H.T, Yscaled[i]); s=w.sum()
    props[i] = w/s if s>0 else w
propdf = pd.DataFrame(props, columns=ref.index, index=expr.columns)
print("\nmean proportions:")
print(propdf.mean().round(3).to_string())

# sanity: tumour purity should be 0.2-0.8, not 0.01 or 0.99
mp = propdf["Malignant"].mean()
flag = "OK" if 0.15 <= mp <= 0.85 else "SUSPICIOUS"
print(f"  -> mean malignant {mp:.3f} [{flag}]")

# ---------- 5. split DEPTOR ----------
dC = ref["DEPTOR"].astype(float)
contrib = pd.DataFrame({c: propdf[c]*dC[c] for c in ref.index}, index=propdf.index)
tot = contrib.sum(axis=1).replace(0,np.nan)
out = pd.DataFrame(index=propdf.index)
out["DEPTOR"] = expr.loc["DEPTOR"] if "DEPTOR" in expr.index else np.nan
out["DEPTOR_tumor_frac"] = contrib["Malignant"]/tot
out["DEPTOR_immune_frac"] = contrib[["Myeloid","Tcell","Bcell","NK"]].sum(axis=1)/tot
out["DEPTOR_myeloid_frac"] = contrib["Myeloid"]/tot
out["purity_marker"] = np.nan
tum_mk = ["KRT5","KRT6A","KRT14","KRT8","KRT18","EPCAM","TP63","KRT19","SFN","KRT17"]
zs=[]
for g in tum_mk:
    if g in expr.index:
        v=expr.loc[g]; zs.append((v-v.mean())/v.std())
if zs: out["purity_marker"]=pd.concat(zs,axis=1).mean(axis=1)
out["hypoxia"] = np.nan
hyp=["HIF1A","CA9","VEGFA","SLC2A1","LDHA","PGK1","ADM","BNIP3","PDK1"]
zs=[]
for g in hyp:
    if g in expr.index:
        v=expr.loc[g]; zs.append((v-v.mean())/v.std())
if zs: out["hypoxia"]=pd.concat(zs,axis=1).mean(axis=1)
out["NK_sig"]=np.nan
nkg=["NKG7","GNLY","KLRD1","PRF1","FCGR3A","NCAM1","KLRF1","XCL1","XCL2","TBX21","EOMES"]
zs=[]
for g in nkg:
    if g in expr.index:
        v=expr.loc[g]; zs.append((v-v.mean())/v.std())
if zs: out["NK_sig"]=pd.concat(zs,axis=1).mean(axis=1)

# merge clinical
for c in ["os","os_event","pfs","pfs_event","hpv_dna","treatment","uicc_stage","tumor_site","gender","age"]:
    if c in clin.columns: out[c]=clin[c].reindex(out.index)
out.to_csv(os.path.join(OUT,"gse65858_deconv_sample.csv"))
print("\nDEPTOR fractions (mean):")
print(out[["DEPTOR_tumor_frac","DEPTOR_immune_frac","DEPTOR_myeloid_frac"]].mean().round(3).to_string())

# ---------- 6. survival ----------
out["os_t"]=pd.to_numeric(out["os"], errors="coerce")
out["os_e"]=pd.to_numeric(out["os_event"], errors="coerce")
out["pfs_t"]=pd.to_numeric(out["pfs"], errors="coerce")
out["pfs_e"]=pd.to_numeric(out["pfs_event"], errors="coerce")
print("\n=== Cox in GSE65858 (whole cohort, per SD) ===")
rows=[]
for var in ["DEPTOR","DEPTOR_tumor_frac","DEPTOR_immune_frac","DEPTOR_myeloid_frac","hypoxia","NK_sig","purity_marker"]:
    for t in ["os","pfs"]:
        d=out.dropna(subset=[f"{t}_t",f"{t}_e",var]).copy()
        if len(d)<50: continue
        d["z"]=(d[var]-d[var].mean())/d[var].std()
        cph=CoxPHFitter()
        try:
            cph.fit(d[[f"{t}_t",f"{t}_e","z"]],duration_col=f"{t}_t",event_col=f"{t}_e")
            s=cph.summary.loc["z"]
            rows.append(dict(var=var,endpoint=t,HR=round(s["exp(coef)"],3),
                             lo=round(s["exp(coef) lower 95%"],3),hi=round(s["exp(coef) upper 95%"],3),
                             p=round(s["p"],4),n=len(d)))
            print(f"  {var:22s} {t:4s} HR/SD={s['exp(coef)']:.2f} ({s['exp(coef) lower 95%']:.2f}-{s['exp(coef) upper 95%']:.2f}) p={s['p']:.3g} n={len(d)}")
        except Exception as e: print("  fail",var,t,str(e)[:40])
pd.DataFrame(rows).to_csv(os.path.join(OUT,"gse65858_cox.csv"),index=False)

# ---------- stratified: treatment & HPV (does not depend on deconvolution) ----------
print("\n=== DEPTOR by treatment and HPV (deconvolution-free) ===")
strat_rows=[]
for strat_col in ["treatment","hpv_dna"]:
    if strat_col not in out.columns: continue
    for val in sorted(out[strat_col].dropna().unique()):
        sub=out[out[strat_col]==val]
        for t in ["os","pfs"]:
            d=sub.dropna(subset=[f"{t}_t",f"{t}_e","DEPTOR"]).copy()
            if len(d)<30: continue
            d["z"]=(d.DEPTOR-d.DEPTOR.mean())/d.DEPTOR.std()
            cph=CoxPHFitter()
            try:
                cph.fit(d[[f"{t}_t",f"{t}_e","z"]],duration_col=f"{t}_t",event_col=f"{t}_e")
                s=cph.summary.loc["z"]
                strat_rows.append(dict(stratum=f"{strat_col}={val}",endpoint=t,n=len(d),
                    HR=round(s["exp(coef)"],3),lo=round(s["exp(coef) lower 95%"],3),
                    hi=round(s["exp(coef) upper 95%"],3),p=round(s["p"],4)))
                print(f"  {strat_col}={val:12s} {t} n={len(d):3d} HR/SD={s['exp(coef)']:.2f} "
                      f"({s['exp(coef) lower 95%']:.2f}-{s['exp(coef) upper 95%']:.2f}) p={s['p']:.3g}")
            except Exception as e: pass
pd.DataFrame(strat_rows).to_csv(os.path.join(OUT,"gse65858_stratified.csv"),index=False)

# treatment interaction
print("\n=== treatment (multi vs mono) interaction ===")
out["multi"]=(out["treatment"].astype(str).str.lower().str.startswith("multi")).astype(int)
for t in ["os","pfs"]:
    d=out.dropna(subset=[f"{t}_t",f"{t}_e","DEPTOR_tumor_frac","multi"]).copy()
    if len(d)<80: continue
    d["fz"]=(d.DEPTOR_tumor_frac-d.DEPTOR_tumor_frac.mean())/d.DEPTOR_tumor_frac.std()
    d["inter"]=d.fz*d.multi
    cph=CoxPHFitter()
    try:
        cph.fit(d[[f"{t}_t",f"{t}_e","fz","multi","inter"]],duration_col=f"{t}_t",event_col=f"{t}_e")
        it=cph.summary.loc["inter"]
        print(f"  {t}: inter HR={it['exp(coef)']:.3f} ({it['exp(coef) lower 95%']:.3f}-{it['exp(coef) upper 95%']:.3f}) p={it['p']:.3g} n={len(d)}")
    except Exception as e: print("  fail",t,str(e)[:50])

# ---------- 7. mediation ----------
print("\n=== Mediation: DEPTOR -> hypoxia -> OS/PFS ===")
med=[]
for X in ["DEPTOR","DEPTOR_tumor_frac"]:
    for t in ["os","pfs"]:
        d=out.dropna(subset=[f"{t}_t",f"{t}_e",X,"hypoxia"]).copy()
        if len(d)<80: continue
        d["Xz"]=(d[X]-d[X].mean())/d[X].std(); d["Mz"]=(d.hypoxia-d.hypoxia.mean())/d.hypoxia.std()
        a=np.polyfit(d.Xz,d.Mz,1)[0]
        c1=CoxPHFitter(); c1.fit(d[[f"{t}_t",f"{t}_e","Xz"]],duration_col=f"{t}_t",event_col=f"{t}_e")
        lc=np.log(c1.summary.loc["Xz","exp(coef)"]); pc=c1.summary.loc["Xz","p"]
        c2=CoxPHFitter(); c2.fit(d[[f"{t}_t",f"{t}_e","Xz","Mz"]],duration_col=f"{t}_t",event_col=f"{t}_e")
        lc2=np.log(c2.summary.loc["Xz","exp(coef)"]); pc2=c2.summary.loc["Xz","p"]
        lb=np.log(c2.summary.loc["Mz","exp(coef)"]); pb=c2.summary.loc["Mz","p"]
        prop=(lc-lc2)/lc if lc not in (0,np.nan) else np.nan
        med.append(dict(X=X,endpoint=t,n=len(d),a=round(a,3),total=round(lc,3),p_total=round(pc,4),
                        direct=round(lc2,3),p_direct=round(pc2,4),b=round(lb,3),p_b=round(pb,4),
                        mediated=round(prop,3)))
        print(f"  X={X:20s} {t} n={len(d)} a={a:+.2f} total={lc:+.3f}(p={pc:.3g}) direct={lc2:+.3f}(p={pc2:.3g}) b={lb:+.3f}(p={pb:.3g}) mediated={prop:.1%}")
pd.DataFrame(med).to_csv(os.path.join(OUT,"gse65858_mediation.csv"),index=False)

# ---------- figures ----------
fig, ax = plt.subplots(figsize=(7,4))
order=["Malignant","Stroma","Endo","Myeloid","Tcell","Bcell","NK"]
ax.bar(order,[propdf[c].mean() for c in order],color="#4c72b0")
ax.set_ylabel("mean proportion"); ax.set_title("GSE65858 deconvolution (GSE164690 full-transcriptome reference)")
plt.xticks(rotation=45,ha="right"); plt.tight_layout()
plt.savefig(os.path.join(OUT,"gse65858_proportions.png"),dpi=140); plt.close()

fig, ax = plt.subplots(figsize=(5.5,4))
ax.boxplot([out.DEPTOR_tumor_frac.dropna(),out.DEPTOR_immune_frac.dropna(),out.DEPTOR_myeloid_frac.dropna()],
           tick_labels=["tumour","immune","myeloid"])
ax.set_ylabel("fraction of bulk DEPTOR"); ax.set_title("GSE65858: bulk DEPTOR contribution")
plt.tight_layout(); plt.savefig(os.path.join(OUT,"gse65858_DEPTOR_fracs.png"),dpi=140); plt.close()

print("\nSaved to", OUT, "DONE.")
