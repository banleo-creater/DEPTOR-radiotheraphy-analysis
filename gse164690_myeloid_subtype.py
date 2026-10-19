#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GSE164690 (Kürten et al. Nat Commun 2021) — MYELOID SUBTYPING extension.
Local source (N: drive). Builds on the dual-axis pipeline and further resolves
the myeloid compartment into:
   - M2-like TAM  : CD163, MRC1, VSIG4, MSR1, IL10RA, CCL2, ARG1, FABP4
   - cDC (DC)     : CD1C, FCER1A, IRF4, CLEC9A, BATF3, XCR1, CCR7, LAMP3, HLA-DRA
   - Neutrophil/PMN (if any): S100A8, S100A9, FCGR3B
Then asks, at the patient level, which myeloid state DEPTOR tracks with,
and whether DEPTOR in malignant cells / M2-TAM / cDC associates with NK abundance.
Also separates Macrophage vs DC DEPTOR and tests NK correlation separately.
"""
import os, re, glob, gzip, warnings
import numpy as np, pandas as pd
import scipy.io
from anndata import AnnData
import scanpy as sc
from scipy.stats import spearmanr, pearsonr
warnings.filterwarnings("ignore"); sc.settings.verbosity = 0

BASE = "WorkBuddy/2026-10-17-11-15-46"
LOCAL = "<PATH>"
OUT  = os.path.join(BASE, "data", "results_myeloid"); os.makedirs(OUT, exist_ok=True)
GENES = ["DEPTOR", "EPYC"]

IMMUNE_MARKERS = {
    "Tcell":  ["CD3D","CD3E","CD8A","CD8B","CD4","IL7R","CD2"],
    "NK":     ["NCAM1","NKG7","KLRF1","KLRD1","GNLY","TBX21","PRF1","GZMB","KLRK1","XCL1","XCL2","FCGR3A"],
    "Bcell":  ["CD79A","CD79B","MS4A1","JCHAIN","MZB1","CD19"],
    "Myeloid":["LYZ","CD68","CD163","CSF1R","FCGR1A","ITGAM","VSIG4","MRC1","HLA-DRA","CD14","SPI1","AIF1","LST1"],
}
# myeloid sub-state markers
M2TAM_MARKERS = ["CD163","VSIG4","MRC1","MSR1","IL10RA","CCL2","ARG1","FABP4","APOE","C1QA","C1QB","SIGLEC1","MARCO"]
CDC_MARKERS   = ["CD1C","FCER1A","IRF4","CLEC9A","BATF3","XCR1","CCR7","LAMP3","HLA-DRA","CD83","CXCL9","CXCL10"]
PMN_MARKERS   = ["S100A8","S100A9","FCGR3B","CXCR2","CEACAM1"]

TUMOR_MARKERS = {
    "Malignant": ["EPCAM","KRT5","KRT14","KRT8","KRT18","TP63","KRT19","SFN","ANXA1"],
    "Stroma":    ["COL1A1","COL1A2","ACTA2","PDGFRA","PDGFRB","DCN","LUM","FAP","MMP2"],
    "Endothelial":["PECAM1","CLDN5","VWF","CDH5"],
}
IMMUNE_LEAK = ["PTPRC","CD3D","CD68","CD163"]

def read_sample(bc, ft, mx):
    with gzip.open(bc, 'rt') as f:
        barcodes = [l.strip() for l in f if l.strip()]
    feat = []
    with gzip.open(ft, 'rt') as f:
        for l in f:
            parts = l.rstrip('\n').split('\t')
            feat.append(parts)
    gene_ids = [r[0] for r in feat]
    gene_symbols = [r[1] if len(r) > 1 else r[0] for r in feat]
    mat = scipy.io.mmread(mx).tocsc()
    ad = AnnData(mat.T.tocsr())
    ad.obs_names = [f"{b}" for b in barcodes]
    ad.var_names = pd.Index(gene_symbols)
    ad.var['gene_ids'] = gene_ids
    return ad

def load_compartment(tag):
    bcs = sorted(glob.glob(os.path.join(LOCAL, f"*_{tag}_barcodes.tsv.gz")))
    ads = []; meta = {}
    for bc in bcs:
        base = bc.replace("_barcodes.tsv.gz", "")
        ft = base + "_features.tsv.gz"; mx = base + "_matrix.mtx.gz"
        m = re.search(r"(HN\d+)_", os.path.basename(base))
        pat = m.group(1) if m else os.path.basename(base)
        ad = read_sample(bc, ft, mx); ad.var_names_make_unique()
        ad.obs["patient"] = pat
        ads.append(ad); meta[os.path.basename(base)] = pat
    comb = sc.concat(ads, join="inner", index_unique="-")
    comb.obs["compartment"] = tag
    sc.pp.filter_cells(comb, min_genes=200)
    sc.pp.normalize_total(comb, target_sum=1e4); sc.pp.log1p(comb)
    return comb, meta

def score(adata, mk):
    vidx = {str(v): i for i, v in enumerate(adata.var_names)}
    idxs = [vidx[m] for m in mk if m in vidx]
    if not idxs: return np.zeros(adata.n_obs)
    sub = adata.X[:, idxs].mean(axis=1)
    if hasattr(sub, "todense"): sub = sub.todense()
    return np.asarray(sub).ravel()

def gvec(adata, g):
    vidx = {str(v): i for i, v in enumerate(adata.var_names)}
    gi = vidx.get(g)
    if gi is None: return None
    col = adata.X[:, gi]
    if hasattr(col, "todense"): col = col.todense()
    return np.asarray(col).ravel()

print("Loading CD45+ / CD45n from local ...")
imm_ad, _ = load_compartment("CD45p")
tn_ad,  _ = load_compartment("CD45n")
for g in GENES:
    print(f"  {g}: CD45p={g in imm_ad.var_names} CD45n={g in tn_ad.var_names}")

# ---- immune lineage ----
smat = np.vstack([score(imm_ad, mk) for mk in IMMUNE_MARKERS.values()]).T
lin = np.array(list(IMMUNE_MARKERS.keys()))[smat.argmax(axis=1)]
lin[smat.max(axis=1) < 0.05] = "Other"
imm_ad.obs["lineage"] = lin
mye = lin == "Myeloid"
sm, sd = score(imm_ad, M2TAM_MARKERS), score(imm_ad, CDC_MARKERS)
spmn = score(imm_ad, PMN_MARKERS)
# resolve: Macrophage (M2-TAM) vs cDC vs PMN
newlin = np.array(lin, dtype=object).copy()   # object dtype -> no string truncation
for i in np.where(mye)[0]:
    scores = {"Macrophage": sm[i], "cDC": sd[i], "PMN": spmn[i]}
    best = max(scores, key=scores.get)
    newlin[i] = best if scores[best] > 0.05 else "Myeloid_other"
imm_ad.obs["myeloid_sub"] = newlin
print("\nImmune lineages (with myeloid split):")
print(imm_ad.obs["myeloid_sub"].value_counts().to_string())

# ---- tumor/stroma ----
epi_s = score(tn_ad, TUMOR_MARKERS["Malignant"]); stro_s = score(tn_ad, TUMOR_MARKERS["Stroma"]); endo_s = score(tn_ad, TUMOR_MARKERS["Endothelial"])
imm_leak = score(tn_ad, IMMUNE_LEAK)
tclass = np.full(tn_ad.n_obs, "Other", dtype=object)
for i in range(tn_ad.n_obs):
    if imm_leak[i] > 0.5: tclass[i] = "ImmuneContam"; continue
    a, b, c = epi_s[i], stro_s[i], endo_s[i]
    if a >= b and a >= c and a > 0.3: tclass[i] = "Malignant"
    elif b >= c and b > 0.3: tclass[i] = "Stroma"
    elif c > 0.3: tclass[i] = "Endothelial"
    else: tclass[i] = "Other"
tn_ad.obs["tumclass"] = tclass
print("\nTumor/stroma:")
print(tn_ad.obs["tumclass"].value_counts().to_string())

imm_lin = imm_ad.obs["myeloid_sub"].values
imm_pat = imm_ad.obs["patient"].values
tn_tc   = tn_ad.obs["tumclass"].values
tn_pat  = tn_ad.obs["patient"].values
imm_vec = {g: gvec(imm_ad, g) for g in GENES}
tn_vec  = {g: gvec(tn_ad, g) for g in GENES}

def aggregate():
    rows = []
    for pat in sorted(set(imm_pat) | set(tn_pat)):
        rec = dict(patient=pat)
        im = imm_pat == pat
        n_imm = int(im.sum())
        n_nk = int(((imm_lin=="NK") & im).sum())
        rec.update(dict(n_immune=n_imm, n_NK=n_nk, NK_fraction=n_nk/n_imm if n_imm else np.nan,
                        n_Macrophage=int(((imm_lin=="Macrophage")&im).sum()),
                        n_cDC=int(((imm_lin=="cDC")&im).sum()),
                        n_PMN=int(((imm_lin=="PMN")&im).sum()),
                        Mac_fraction=((imm_lin=="Macrophage")&im).sum()/n_imm if n_imm else np.nan,
                        cDC_fraction=((imm_lin=="cDC")&im).sum()/n_imm if n_imm else np.nan))
        tm = tn_pat == pat
        rec.update(dict(n_malignant=int(((tn_tc=="Malignant")&tm).sum()),
                        n_stroma=int(((tn_tc=="Stroma")&tm).sum())))
        nk_m  = (imm_lin=="NK") & im
        mac_m = (imm_lin=="Macrophage") & im
        cdc_m = (imm_lin=="cDC") & im
        mye_m = np.isin(imm_lin, ["Macrophage","cDC","PMN","Myeloid_other"]) & im
        mal_m = (tn_tc=="Malignant") & tm
        str_m = (tn_tc=="Stroma") & tm
        for g in GENES:
            iv, tv = imm_vec[g], tn_vec[g]
            if iv is not None:
                rec[f"{g}_NK_mean"]       = float(iv[nk_m].mean()) if nk_m.sum() else np.nan
                rec[f"{g}_Mac_mean"]      = float(iv[mac_m].mean()) if mac_m.sum() else np.nan
                rec[f"{g}_Mac_detect"]    = float((iv[mac_m]>0).mean()) if mac_m.sum() else np.nan
                rec[f"{g}_cDC_mean"]      = float(iv[cdc_m].mean()) if cdc_m.sum() else np.nan
                rec[f"{g}_cDC_detect"]    = float((iv[cdc_m]>0).mean()) if cdc_m.sum() else np.nan
                rec[f"{g}_myeloid_mean"]  = float(iv[mye_m].mean()) if mye_m.sum() else np.nan
            if tv is not None:
                rec[f"{g}_malignant_mean"]  = float(tv[mal_m].mean()) if mal_m.sum() else np.nan
                rec[f"{g}_malignant_detect"]= float((tv[mal_m]>0).mean()) if mal_m.sum() else np.nan
                rec[f"{g}_stroma_mean"]     = float(tv[str_m].mean()) if str_m.sum() else np.nan
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("patient").reset_index(drop=True)

df = aggregate()
df.to_csv(os.path.join(OUT, "myeloid_subtype_summary.csv"), index=False)
print("\n=== Patient summary ===")
print(df[["patient","n_immune","n_NK","NK_fraction","n_Macrophage","n_cDC",
          "DEPTOR_malignant_mean","DEPTOR_Mac_mean","DEPTOR_cDC_mean","DEPTOR_myeloid_mean",
          "EPYC_Mac_mean","EPYC_cDC_mean"]].to_string(index=False))

# ---- correlations ----
print("\n=== Myeloid-subtype DEPTOR correlations (patient level) ===")
corr_rows = []
def add_corr(a, b, desc, g):
    s = df[[a,b]].dropna()
    if len(s) >= 4:
        rho,p = spearmanr(s[a], s[b]); r,p2 = pearsonr(s[a], s[b])
        corr_rows.append(dict(gene=g, comparison=desc, n=len(s),
            Spearman_rho=round(rho,3), Spearman_p=round(p,4),
            Pearson_r=round(r,3), Pearson_p=round(p2,4)))
        print(f"  [{g}] {desc}: rho={rho:+.3f} (p={p:.3g}) r={r:+.3f} (p={p2:.3g}) n={len(s)}")
for g in GENES:
    add_corr(f"{g}_malignant_mean","NK_fraction", "tumor DEPTOR vs NK fraction", g)
    add_corr(f"{g}_Mac_mean","NK_fraction", "M2-TAM DEPTOR vs NK fraction", g)
    add_corr(f"{g}_cDC_mean","NK_fraction", "cDC DEPTOR vs NK fraction", g)
    add_corr(f"{g}_myeloid_mean","NK_fraction", "total myeloid DEPTOR vs NK fraction", g)
    add_corr(f"{g}_malignant_mean",f"{g}_Mac_mean", "tumor DEPTOR vs M2-TAM DEPTOR", g)
    add_corr(f"{g}_malignant_mean",f"{g}_cDC_mean", "tumor DEPTOR vs cDC DEPTOR", g)
    add_corr(f"{g}_Mac_mean",f"{g}_cDC_mean", "M2-TAM DEPTOR vs cDC DEPTOR", g)
pd.DataFrame(corr_rows).to_csv(os.path.join(OUT,"myeloid_subtype_correlations.csv"), index=False)

# ---- figures ----
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rng = np.random.default_rng(1)
def umap_fig(adata,color,title,fname):
    n=min(adata.n_obs,30000)
    vis=adata[adata.obs_names.values[rng.choice(adata.n_obs,n,replace=False)]].copy()
    sc.pp.neighbors(vis,n_neighbors=15,n_pcs=30); sc.tl.umap(vis)
    plt.figure(figsize=(7,6)); sc.pl.umap(vis,color=color,show=False,title=title,ax=plt.gca())
    plt.tight_layout(); plt.savefig(os.path.join(OUT,fname),dpi=150); plt.close()
umap_fig(imm_ad,"myeloid_sub","CD45+ myeloid subtypes","myeloid_sub_umap_immune.png")
umap_fig(tn_ad,"tumclass","CD45n tumor/stroma","myeloid_sub_umap_tumor.png")

def scat(x,y,df,title,fname,color):
    s=df[[x,y]].dropna()
    if len(s)<4: return
    rho,p=spearmanr(s[x],s[y])
    plt.figure(figsize=(6,5))
    plt.scatter(s[x],s[y],c=color,s=60,edgecolor="k")
    for _,rr in df.iterrows():
        if pd.notna(rr[x]) and pd.notna(rr[y]):
            plt.annotate(rr.patient,(rr[x],rr[y]),fontsize=7,xytext=(3,3),textcoords="offset points")
    plt.xlabel(x); plt.ylabel(y); plt.title(f"{title}\nSpearman rho={rho:+.2f} (p={p:.2g})")
    plt.tight_layout(); plt.savefig(os.path.join(OUT,fname),dpi=150); plt.close()

scat("DEPTOR_Mac_mean","NK_fraction",df,"M2-TAM DEPTOR vs NK fraction","myeloid_MacDEPTOR_vs_NK.png","#d62728")
scat("DEPTOR_cDC_mean","NK_fraction",df,"cDC DEPTOR vs NK fraction","myeloid_cDCDEPTOR_vs_NK.png","#9467bd")
scat("DEPTOR_malignant_mean","NK_fraction",df,"Tumor DEPTOR vs NK fraction","myeloid_tumorDEPTOR_vs_NK.png","#2ca02c")
scat("DEPTOR_Mac_mean","DEPTOR_malignant_mean",df,"M2-TAM DEPTOR vs Tumor DEPTOR","myeloid_Mac_vs_tumor_DEPTOR.png","#ff7f0e")

print("\nSaved to", OUT, "DONE.")
