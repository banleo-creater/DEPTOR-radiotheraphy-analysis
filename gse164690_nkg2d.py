#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GSE164690 — cell-level test of the NKG2D-ligand hypothesis on top of the
myeloid DEPTOR -> NK axis.

Idea:
    myeloid DEPTOR high -> few NK (already shown, rho=-0.68)
    radiotherapy up-regulates NKG2D ligands on tumour cells
    -> if NK are few, ligand up-regulation has "no receiver"

Here we can measure, per patient:
  - myeloid DEPTOR            (CD45+ compartment)
  - NK fraction               (CD45+ compartment)
  - tumour NKG2D-ligand score (CD45n malignant cells: MICA, MICB, ULBP1-3,
                               RAET1E/G/L = ULBP4/5/6)
and ask whether NK-poor tumours show compensatory ligand up-regulation,
and how that relates to myeloid DEPTOR.
"""
import os, re, glob, gzip, warnings
import numpy as np, pandas as pd
import scipy.io
from anndata import AnnData
import scanpy as sc
from scipy.stats import spearmanr, pearsonr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore"); sc.settings.verbosity = 0

BASE = "WorkBuddy/2026-10-17-11-15-46"
LOCAL = "<PATH>"
OUT = os.path.join(BASE, "data", "results_nkg2d"); os.makedirs(OUT, exist_ok=True)

LIG = ["MICA","MICB","ULBP1","ULBP2","ULBP3","RAET1E","RAET1G","RAET1L"]

IMMUNE_MARKERS = {
    "Tcell":  ["CD3D","CD3E","CD8A","CD8B","CD4","IL7R","CD2"],
    "NK":     ["NCAM1","NKG7","KLRF1","KLRD1","GNLY","TBX21","PRF1","GZMB","KLRK1","XCL1","XCL2","FCGR3A"],
    "Bcell":  ["CD79A","CD79B","MS4A1","JCHAIN","MZB1","CD19"],
    "Myeloid":["LYZ","CD68","CD163","CSF1R","FCGR1A","ITGAM","VSIG4","MRC1","HLA-DRA","CD14","SPI1","AIF1","LST1"],
}
TUMOR_MARKERS = {
    "Malignant": ["EPCAM","KRT5","KRT14","KRT8","KRT18","TP63","KRT19","SFN","ANXA1"],
    "Stroma":    ["COL1A1","COL1A2","ACTA2","PDGFRA","PDGFRB","DCN","LUM","FAP","MMP2"],
    "Endothelial":["PECAM1","CLDN5","VWF","CDH5"],
}
IMMUNE_LEAK = ["PTPRC","CD3D","CD68","CD163"]

def read_sample(bc, ft, mx):
    with gzip.open(bc, "rt") as f:
        barcodes = [l.strip() for l in f if l.strip()]
    feat = []
    with gzip.open(ft, "rt") as f:
        for l in f:
            feat.append(l.rstrip("\n").split("\t"))
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

print("Loading CD45p / CD45n ...")
imm = load_comp("CD45p"); tn = load_comp("CD45n")

# immune lineages
S = np.vstack([score(imm, mk) for mk in IMMUNE_MARKERS.values()]).T
lin = np.array(list(IMMUNE_MARKERS.keys()), dtype=object)[S.argmax(axis=1)]
lin[S.max(axis=1) < 0.05] = "Other"
imm.obs["lin"] = lin

# tumor classification
epi = score(tn, TUMOR_MARKERS["Malignant"]); str_ = score(tn, TUMOR_MARKERS["Stroma"])
end = score(tn, TUMOR_MARKERS["Endothelial"]); leak = score(tn, IMMUNE_LEAK)
tc = np.full(tn.n_obs, "Other", dtype=object)
for i in range(tn.n_obs):
    if leak[i] > 0.5: tc[i] = "ImmuneContam"; continue
    a, b, c = epi[i], str_[i], end[i]
    if a >= b and a >= c and a > 0.3: tc[i] = "Malignant"
    elif b >= c and b > 0.3: tc[i] = "Stroma"
    elif c > 0.3: tc[i] = "Endothelial"
tn.obs["tc"] = tc
print("immune lineages:", pd.Series(lin).value_counts().to_dict())
print("tumor classes:", pd.Series(tc).value_counts().to_dict())

# gene vectors
def gvec(ad, g):
    vidx = {str(v): i for i, v in enumerate(ad.var_names)}
    gi = vidx.get(g)
    if gi is None: return None
    c = ad.X[:, gi]
    return np.asarray(c.todense()).ravel() if hasattr(c, "todense") else np.asarray(c).ravel()

ivD = gvec(imm, "DEPTOR")
# malignant NKG2D-ligand score per cell
lig_idx = [i for i, v in enumerate(tn.var_names) if str(v) in LIG]
print("ligand genes present in CD45n:", [str(tn.var_names[i]) for i in lig_idx])
lig_cell = np.asarray(tn.X[:, lig_idx].mean(axis=1)).ravel() if lig_idx else np.zeros(tn.n_obs)

imm_lin = imm.obs["lin"].values; imm_pat = imm.obs["patient"].values
tn_tc = tn.obs["tc"].values; tn_pat = tn.obs["patient"].values
mye_cls = np.isin(imm_lin, ["Myeloid"])

rows = []
for pat in sorted(set(imm_pat) | set(tn_pat)):
    im = imm_pat == pat; tm = tn_pat == pat
    n_imm = int(im.sum()); n_nk = int(((imm_lin == "NK") & im).sum())
    mal = (tn_tc == "Malignant") & tm
    rec = dict(patient=pat, n_immune=n_imm, n_NK=n_nk,
               NK_fraction=n_nk / n_imm if n_imm else np.nan,
               n_malignant=int(mal.sum()))
    rec["myeloid_DEPTOR"] = float(ivD[mye_cls & im].mean()) if (mye_cls & im).sum() else np.nan
    rec["tumor_NKG2D_ligand"] = float(lig_cell[mal].mean()) if mal.sum() else np.nan
    rec["tumor_ligand_detect"] = float((lig_cell[mal] > 0).mean()) if mal.sum() else np.nan
    rows.append(rec)
df = pd.DataFrame(rows).sort_values("patient").reset_index(drop=True)
df.to_csv(os.path.join(OUT, "gse164690_nkg2d_patient.csv"), index=False)
print("\n=== patient table ===")
print(df.to_string(index=False))

print("\n=== correlations (patient level) ===")
pairs = [("myeloid_DEPTOR","NK_fraction","myeloid DEPTOR vs NK fraction"),
         ("tumor_NKG2D_ligand","NK_fraction","tumor NKG2D ligand vs NK fraction"),
         ("myeloid_DEPTOR","tumor_NKG2D_ligand","myeloid DEPTOR vs tumor ligand")]
for a, b, desc in pairs:
    s = df[[a, b]].dropna()
    if len(s) >= 5:
        rho, p = spearmanr(s[a], s[b]); r, pr = pearsonr(s[a], s[b])
        print(f"  {desc:42s} n={len(s)} rho={rho:+.3f} (p={p:.3g}) r={r:+.3f} (p={pr:.3g})")

# figure
fig, axes = plt.subplots(1, 3, figsize=(14,4.2))
for ax, (a, b, ttl) in zip(axes, pairs):
    s = df[[a, b]].dropna()
    ax.scatter(s[a], s[b], s=55, edgecolor="k", alpha=.85)
    if len(s) >= 5:
        rho, p = spearmanr(s[a], s[b])
        ax.set_title(f"{ttl}\nrho={rho:+.2f} (p={p:.2g})", fontsize=8)
    ax.set_xlabel(a); ax.set_ylabel(b)
plt.suptitle("GSE164690: myeloid DEPTOR / NK / tumour NKG2D ligands", fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "gse164690_nkg2d_triple.png"), dpi=140); plt.close()

print("\nSaved to", OUT, "DONE.")
