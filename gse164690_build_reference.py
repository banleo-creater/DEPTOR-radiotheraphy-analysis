#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build a cell-type reference matrix from GSE164690 scRNA (CD45+ immune + CD45n
tumour/stroma) over the 67-gene panel, to deconvolve TCGA bulk.

Outputs data/gse164690_reference.csv :
  rows = cell types (Malignant, Myeloid, Tcell, Bcell, NK, Stroma, Endo)
  cols = the 67 genes + DEPTOR  (mean log1p CP10K expression = centroid)
Also saves per-type DEPTOR centroid separately.
"""
import os, re, glob, gzip, warnings
import numpy as np, pandas as pd
import scipy.io
from anndata import AnnData
import scanpy as sc
warnings.filterwarnings("ignore"); sc.settings.verbosity = 0

BASE = "WorkBuddy/2026-10-17-11-15-46"
LOCAL = "<PATH>"
OUT = os.path.join(BASE, "data")

SETS = {
 "Tumor":  ["KRT5","KRT6A","KRT14","KRT8","KRT18","EPCAM","TP63","KRT19","SFN","KRT17"],
 "Myeloid":["LYZ","CD68","CD163","CSF1R","CD14","ITGAM","MRC1","VSIG4","AIF1","C1QA","C1QB",
            "FCGR1A","SPI1","LST1","TYROBP"],
 "Tcell":  ["CD3D","CD3E","CD8A","CD4","IL7R","CD2","CD3G","CD28"],
 "Bcell":  ["CD79A","MS4A1","JCHAIN","MZB1","CD19"],
 "NK":     ["NCAM1","NKG7","KLRF1","KLRD1","GNLY","KLRK1","PRF1"],
 "Stroma": ["COL1A1","COL1A2","ACTA2","PDGFRA","DCN","LUM","FAP","PDGFRB"],
 "Endo":   ["PECAM1","VWF","CLDN5","CDH5"],
}
PANEL = sorted({g for v in SETS.values() for g in v} | {"DEPTOR"})

IMMUNE_MK = {
 "Tcell": SETS["Tcell"], "NK": SETS["NK"], "Bcell": SETS["Bcell"], "Myeloid": SETS["Myeloid"],
}
TUMOR_MK = {"Malignant": SETS["Tumor"], "Stroma": SETS["Stroma"], "Endothelial": SETS["Endo"]}
LEAK = ["PTPRC","CD3D","CD68","CD163"]

def read_sample(bc, ft, mx):
    with gzip.open(bc, "rt") as f:
        barcodes = [l.strip() for l in f if l.strip()]
    feat = []
    with gzip.open(ft, "rt") as f:
        for l in f: feat.append(l.rstrip("\n").split("\t"))
    ids = [r[0] for r in feat]; syms = [r[1] if len(r) > 1 else r[0] for r in feat]
    mat = scipy.io.mmread(mx).tocsc()
    ad = AnnData(mat.T.tocsr())
    ad.obs_names = barcodes; ad.var_names = pd.Index(syms); ad.var["gene_ids"] = ids
    return ad

def load_comp(tag):
    ads = []
    for bc in sorted(glob.glob(os.path.join(LOCAL, f"*_{tag}_barcodes.tsv.gz"))):
        base = bc.replace("_barcodes.tsv.gz", "")
        m = re.search(r"(HN\d+)_", os.path.basename(base))
        if not m: continue
        ad = read_sample(bc, base + "_features.tsv.gz", base + "_matrix.mtx.gz")
        ad.var_names_make_unique(); ad.obs["patient"] = m.group(1); ads.append(ad)
    comb = sc.concat(ads, join="inner", index_unique="-")
    sc.pp.filter_cells(comb, min_genes=200)
    sc.pp.normalize_total(comb, target_sum=1e4); sc.pp.log1p(comb)
    return comb

def score(ad, mk):
    vidx = {str(v): i for i, v in enumerate(ad.var_names)}
    idxs = [vidx[m] for m in mk if m in vidx]
    if not idxs: return np.zeros(ad.n_obs)
    s = ad.X[:, idxs].mean(axis=1)
    return np.asarray(s.todense()).ravel() if hasattr(s, "todense") else np.asarray(s).ravel()

print("loading ...")
imm = load_comp("CD45p"); tn = load_comp("CD45n")

S = np.vstack([score(imm, mk) for mk in IMMUNE_MK.values()]).T
lin = np.array(list(IMMUNE_MK.keys()), dtype=object)[S.argmax(axis=1)]
lin[S.max(axis=1) < 0.05] = "Other"
imm.obs["celltype"] = lin

epi = score(tn, TUMOR_MK["Malignant"]); str_ = score(tn, TUMOR_MK["Stroma"])
end = score(tn, TUMOR_MK["Endothelial"]); leak = score(tn, LEAK)
tc = np.full(tn.n_obs, "Other", dtype=object)
for i in range(tn.n_obs):
    if leak[i] > 0.5: tc[i] = "Other"; continue
    a, b, c = epi[i], str_[i], end[i]
    if a >= b and a >= c and a > 0.3: tc[i] = "Malignant"
    elif b >= c and b > 0.3: tc[i] = "Stroma"
    elif c > 0.3: tc[i] = "Endo"
tn.obs["celltype"] = tc

# combined centroid over PANEL (genes present in both compartments)
def centroid(ad, label):
    m = ad.obs["celltype"].values == label
    if m.sum() == 0: return None
    vidx = {str(v): i for i, v in enumerate(ad.var_names)}
    idxs = [vidx[g] for g in PANEL if g in vidx]
    names = [g for g in PANEL if g in vidx]
    X = ad.X[m][:, idxs]
    mu = np.asarray(X.mean(axis=0)).ravel()
    return pd.Series(mu, index=names)

rows = {}
contrib = {}
for lab, ad in [("Malignant", tn), ("Stroma", tn), ("Endo", tn),
                ("Tcell", imm), ("NK", imm), ("Bcell", imm), ("Myeloid", imm)]:
    s = centroid(ad, lab)
    if s is None: 
        print("  skip", lab); continue
    rows[lab] = s
    contrib[lab] = int((ad.obs["celltype"] == lab).sum())
    print(f"  {lab}: {contrib[lab]} cells, {len(s)} genes")

ref = pd.DataFrame(rows).T
# fill missing genes with 0 (gene absent in that compartment's matrix)
ref = ref.reindex(columns=PANEL).fillna(0.0)
ref.to_csv(os.path.join(OUT, "gse164690_reference.csv"))
print("\nsaved reference:", ref.shape)
print("DEPTOR centroids:")
print(ref["DEPTOR"].to_string())
