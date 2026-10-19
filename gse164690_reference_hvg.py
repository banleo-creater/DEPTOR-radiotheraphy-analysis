#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build a FULL-TRANSCRIPTOME cell-type reference from GSE164690 (CD45+ immune +
CD45n tumour/stroma) and select informative genes (HVG / marker union).

The previous 57-gene reference made NNLS unstable (all-malignant or all-equal).
Using ~1000 informative genes should fix that.

Outputs
  data/gse164690_centroid_full.csv  : 7 cell types x N genes (mean log1p CP10K)
  data/gse164690_hvg.txt            : informative gene list (union of top markers)
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

IMMUNE_MK = {
 "Tcell": ["CD3D","CD3E","CD8A","CD8B","CD4","IL7R","CD2"],
 "NK":    ["NCAM1","NKG7","KLRF1","KLRD1","GNLY","TBX21","PRF1","GZMB","KLRK1","XCL1","XCL2","FCGR3A"],
 "Bcell": ["CD79A","CD79B","MS4A1","JCHAIN","MZB1","CD19"],
 "Myeloid":["LYZ","CD68","CD163","CSF1R","FCGR1A","ITGAM","VSIG4","MRC1","HLA-DRA","CD14","SPI1","AIF1","LST1"],
}
TUMOR_MK = {"Malignant":["EPCAM","KRT5","KRT14","KRT8","KRT18","TP63","KRT19","SFN","ANXA1"],
            "Stroma":["COL1A1","COL1A2","ACTA2","PDGFRA","PDGFRB","DCN","LUM","FAP","MMP2"],
            "Endo":["PECAM1","CLDN5","VWF","CDH5"]}
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
        ad.var_names_make_unique(); ad.obs["patient"] = m.group(1); ad.obs["comp"] = tag
        ads.append(ad)
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

print("loading CD45p / CD45n ...")
imm = load_comp("CD45p"); tn = load_comp("CD45n")

S = np.vstack([score(imm, mk) for mk in IMMUNE_MK.values()]).T
lin = np.array(list(IMMUNE_MK.keys()), dtype=object)[S.argmax(axis=1)]
lin[S.max(axis=1) < 0.05] = "Other"
imm.obs["celltype"] = lin

epi = score(tn, TUMOR_MK["Malignant"]); st = score(tn, TUMOR_MK["Stroma"])
en = score(tn, TUMOR_MK["Endo"]); lk = score(tn, LEAK)
tc = np.full(tn.n_obs, "Other", dtype=object)
for i in range(tn.n_obs):
    if lk[i] > 0.5: continue
    a, b, c = epi[i], st[i], en[i]
    if a >= b and a >= c and a > 0.3: tc[i] = "Malignant"
    elif b >= c and b > 0.3: tc[i] = "Stroma"
    elif c > 0.3: tc[i] = "Endo"
tn.obs["celltype"] = tc

# combined matrix (inner join genes)
comb = sc.concat([imm, tn], join="inner", index_unique="-", merge="same")
genes = np.array([str(g) for g in comb.var_names])
print(f"combined: {comb.n_obs} cells x {len(genes)} genes")

ct = comb.obs["celltype"].values
classes = ["Malignant","Stroma","Endo","Tcell","NK","Bcell","Myeloid"]
# centroid per class
cent = {}
for c in classes:
    m = ct == c
    if m.sum() < 20: continue
    Xc = comb.X[m]
    mu = np.asarray(Xc.mean(axis=0)).ravel()
    cent[c] = mu
    print(f"  {c}: {int(m.sum())} cells")
ref = pd.DataFrame(cent, index=genes).T
ref.to_csv(os.path.join(OUT, "gse164690_centroid_full.csv"))
print("centroid saved:", ref.shape)

# informative genes: top markers per class (mean diff / sd) -> union
means = {c: np.asarray(comb.X[ct == c].mean(axis=0)).ravel() for c in cent}
_mean_all = np.asarray(comb.X.mean(axis=0)).ravel()
_sq = np.asarray(comb.X.multiply(comb.X).mean(axis=0)).ravel()
sds = np.sqrt(np.maximum(_sq - _mean_all**2, 0)) + 1e-6
genes_use = []
for c in cent:
    others = np.mean([means[o] for o in cent if o != c], axis=0)
    score_c = (means[c] - others) / sds
    top = np.argsort(score_c)[-250:]
    genes_use += list(genes[top])
genes_use = sorted(set(genes_use) - {"", "nan"})
# also require gene detected in >=2% of cells overall
detect = np.asarray((comb.X > 0).mean(axis=0)).ravel()
dmap = dict(zip(genes, detect))
genes_use = [g for g in genes_use if dmap.get(g, 0) >= 0.02]
with open(os.path.join(OUT, "gse164690_hvg.txt"), "w") as f:
    f.write("\n".join(genes_use))
print(f"informative genes: {len(genes_use)}")
print("  examples:", genes_use[:15])
