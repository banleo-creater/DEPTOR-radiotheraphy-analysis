#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GSE164690 (Kürten et al. Nat Commun 2021) — DUAL-AXIS pilot, LOCAL source.
Reads directly from the local extracted GEO raw files (GSM-prefixed 10x mtx),
no NCBI download required.

Combines tumor-infiltrating immune (CD45+) and tumor+stroma (CD45n) 10x samples.
Replicates cell types de novo, then asks:
  (A) DEPTOR in malignant (tumor) cells  vs  DEPTOR in myeloid (immune) cells
      across patients -> are they anti-correlated? (the "tumor vs immune" dual axis)
  (B) DEPTOR in malignant / myeloid  vs  NK abundance in CD45+.
EPYC kept for completeness (earlier found near-absent in myeloid).
"""
import os, re, glob, gzip, warnings
import numpy as np, pandas as pd
import scipy.io
from anndata import AnnData
import scanpy as sc
warnings.filterwarnings("ignore"); sc.settings.verbosity = 0

BASE = "WorkBuddy/2026-10-17-11-15-46"
LOCAL = "<PATH>"
OUT  = os.path.join(BASE, "data", "results_dual"); os.makedirs(OUT, exist_ok=True)
GENES = ["EPYC", "DEPTOR"]

IMMUNE_MARKERS = {
    "Tcell":  ["CD3D","CD3E","CD8A","CD8B","CD4","IL7R","CD2"],
    "NK":     ["NCAM1","NKG7","KLRF1","KLRD1","FCGR3A","GNLY","TBX21",
               "PRF1","GZMB","KLRK1","XCL1","XCL2","FCGR3B"],
    "Bcell":  ["CD79A","CD79B","MS4A1","JCHAIN","MZB1","CD19"],
    "Myeloid":["LYZ","CD68","CD163","CSF1R","FCGR1A","ITGAM","VSIG4",
               "MRC1","HLA-DRA","CD14","SPI1","AIF1","LST1"],
}
MAC_MARKERS = ["CD163","VSIG4","MRC1","CD14","FCGR1A","SPI1","APOE","FABP4","C1QA","C1QB"]
DC_MARKERS  = ["FCER1A","IRF8","CLEC9A","CD1C","LILRA4","CCR7","BATF3"]

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
    gene_ids     = [r[0] for r in feat]
    gene_symbols = [r[1] if len(r) > 1 else r[0] for r in feat]
    mat = scipy.io.mmread(mx).tocsc()
    ad = AnnData(mat.T.tocsr())            # cells x genes
    ad.obs_names = [f"{b}" for b in barcodes]
    ad.var_names = pd.Index(gene_symbols)
    ad.var['gene_ids'] = gene_ids
    return ad

def load_compartment(tag):
    """tag = 'CD45p' or 'CD45n'; returns (anndata, dict sample->patient)."""
    bcs = sorted(glob.glob(os.path.join(LOCAL, f"*_{tag}_barcodes.tsv.gz")))
    ads = []
    meta = {}
    for bc in bcs:
        base = bc.replace("_barcodes.tsv.gz", "")
        ft = base + "_features.tsv.gz"
        mx = base + "_matrix.mtx.gz"
        m = re.search(r"(HN\d+)_", os.path.basename(base))
        pat = m.group(1) if m else os.path.basename(base)
        ad = read_sample(bc, ft, mx)
        ad.var_names_make_unique()
        ad.obs["patient"] = pat
        ads.append(ad); meta[os.path.basename(base)] = pat
        print(f"  loaded {os.path.basename(base)}: {ad.n_obs} cells")
    if not ads:
        raise SystemExit(f"No {tag} samples found.")
    comb = sc.concat(ads, join="inner", index_unique="-")
    comb.obs["compartment"] = tag
    print(f"  {tag} combined: {comb.n_obs} cells x {comb.n_vars} genes")
    sc.pp.filter_cells(comb, min_genes=200)
    sc.pp.normalize_total(comb, target_sum=1e4); sc.pp.log1p(comb)
    return comb, meta

def score(adata, mk):
    vidx = {str(v): i for i, v in enumerate(adata.var_names)}
    idxs = [vidx[m] for m in mk if m in vidx]
    if not idxs:
        return np.zeros(adata.n_obs)
    sub = adata.X[:, idxs].mean(axis=1)
    if hasattr(sub, "todense"):
        sub = sub.todense()
    return np.asarray(sub).ravel()

print("="*70)
print("Reading CD45+ (immune) and CD45n (tumor+stroma) from LOCAL source ...")
print("="*70)
imm_ad, _ = load_compartment("CD45p")
tn_ad,  _ = load_compartment("CD45n")

for g in GENES:
    print(f"  {g} in CD45p: {g in imm_ad.var_names} | {g} in CD45n: {g in tn_ad.var_names}")

# ---------------- CD45p immune lineage ----------------
print("\nClassifying CD45+ immune lineages ...")
smat = np.vstack([score(imm_ad, mk) for mk in IMMUNE_MARKERS.values()]).T
lin = np.array(list(IMMUNE_MARKERS.keys()))[smat.argmax(axis=1)]
lin[smat.max(axis=1) < 0.05] = "Other"
imm_ad.obs["lineage"] = lin
mye = imm_ad.obs["lineage"].values == "Myeloid"
sm, sd = score(imm_ad, MAC_MARKERS), score(imm_ad, DC_MARKERS)
imm_ad.obs.loc[imm_ad.obs.index[mye], "lineage"] = np.where(sm[mye] >= sd[mye], "Macrophage", "DC")
print(imm_ad.obs["lineage"].value_counts().to_string())

# ---------------- CD45n tumor/stroma ----------------
print("\nClassifying CD45n tumor/stroma ...")
epi_s  = score(tn_ad, TUMOR_MARKERS["Malignant"])
stro_s = score(tn_ad, TUMOR_MARKERS["Stroma"])
endo_s = score(tn_ad, TUMOR_MARKERS["Endothelial"])
imm_leak = score(tn_ad, IMMUNE_LEAK)
tclass = np.full(tn_ad.n_obs, "Other", dtype=object)
for i in range(tn_ad.n_obs):
    if imm_leak[i] > 0.5:
        tclass[i] = "ImmuneContam"; continue
    a, b, c = epi_s[i], stro_s[i], endo_s[i]
    if a >= b and a >= c and a > 0.3: tclass[i] = "Malignant"
    elif b >= c and b > 0.3:          tclass[i] = "Stroma"
    elif c > 0.3:                     tclass[i] = "Endothelial"
    else:                             tclass[i] = "Other"
tn_ad.obs["tumclass"] = tclass
print(tn_ad.obs["tumclass"].value_counts().to_string())

# ---------------- gene vectors ----------------
def gvec(adata, g):
    vidx = {str(v): i for i, v in enumerate(adata.var_names)}
    gi = vidx.get(g)
    if gi is None:
        return None
    col = adata.X[:, gi]
    if hasattr(col, "todense"):
        col = col.todense()
    return np.asarray(col).ravel()

imm_vec = {g: gvec(imm_ad, g) for g in GENES}
tn_vec  = {g: gvec(tn_ad, g) for g in GENES}
imm_lin = imm_ad.obs["lineage"].values
imm_pat = imm_ad.obs["patient"].values
tn_tc   = tn_ad.obs["tumclass"].values
tn_pat  = tn_ad.obs["patient"].values
mye_classes = np.isin(imm_lin, ["Myeloid","Macrophage","DC"])

# ---------------- per-patient aggregates ----------------
def aggregate():
    rows = []
    all_patients = sorted(set(imm_pat) | set(tn_pat))
    for pat in all_patients:
        rec = dict(patient=pat)
        # immune side
        im = imm_pat == pat
        n_imm = int(im.sum())
        n_nk  = int(((imm_lin== "NK") & im).sum())
        n_mye = int((mye_classes & im).sum())
        rec.update(dict(n_immune=n_imm, n_NK=n_nk, n_myeloid=n_mye,
                        NK_fraction_of_immune=(n_nk/n_imm if n_imm else np.nan)))
        # tumor side
        tm = tn_pat == pat
        rec.update(dict(n_malignant=int(((tn_tc=="Malignant") & tm).sum()),
                        n_stroma=int(((tn_tc=="Stroma") & tm).sum()),
                        n_endothelial=int(((tn_tc=="Endothelial") & tm).sum())))
        for g in GENES:
            iv, tv = imm_vec[g], tn_vec[g]
            if iv is not None:
                rec[f"{g}_NK_mean"]      = float(iv[(imm_lin=="NK") & im].mean())
                rec[f"{g}_myeloid_mean"] = float(iv[mye_classes & im].mean())
                rec[f"{g}_myeloid_detect"]=float((iv[mye_classes & im]>0).mean())
            else:
                rec[f"{g}_NK_mean"]=rec[f"{g}_myeloid_mean"]=rec[f"{g}_myeloid_detect"]=np.nan
            if tv is not None:
                rec[f"{g}_malignant_mean"] = float(tv[(tn_tc=="Malignant") & tm].mean())
                rec[f"{g}_malignant_detect"]=float((tv[(tn_tc=="Malignant") & tm]>0).mean())
                rec[f"{g}_stroma_mean"]    = float(tv[(tn_tc=="Stroma") & tm].mean())
                rec[f"{g}_stroma_detect"]  = float((tv[(tn_tc=="Stroma") & tm]>0).mean())
            else:
                for s in ["malignant","stroma"]:
                    rec[f"{g}_{s}_mean"]=rec[f"{g}_{s}_detect"]=np.nan
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("patient").reset_index(drop=True)

df = aggregate()
df.to_csv(os.path.join(OUT, "dual_axis_summary.csv"), index=False)
print("\n=== Patient-level dual-axis summary ===")
cols = ["patient","n_immune","n_NK","NK_fraction_of_immune","n_malignant","n_stroma",
        "DEPTOR_malignant_mean","DEPTOR_malignant_detect","DEPTOR_stroma_mean",
        "DEPTOR_myeloid_mean","DEPTOR_myeloid_detect","DEPTOR_NK_mean",
        "EPYC_malignant_mean","EPYC_myeloid_mean","EPYC_myeloid_detect"]
print(df[cols].to_string(index=False))

# ---------------- correlations (the dual axis) ----------------
from scipy.stats import spearmanr, pearsonr
print("\n=== Dual-axis correlations (patient level) ===")
corr_rows = []
for g in GENES:
    pairs = [
        (f"{g}_malignant_mean", f"{g}_myeloid_mean",   "tumor DEPTOR vs myeloid DEPTOR"),
        (f"{g}_malignant_mean", "NK_fraction_of_immune","tumor DEPTOR vs NK fraction"),
        (f"{g}_myeloid_mean",   "NK_fraction_of_immune","myeloid DEPTOR vs NK fraction"),
        (f"{g}_malignant_mean", f"{g}_stroma_mean",     "tumor DEPTOR vs stroma DEPTOR"),
    ]
    for a, b, desc in pairs:
        s = df[[a, b]].dropna()
        if len(s) >= 4:
            rho, p = spearmanr(s[a], s[b]); r, p2 = pearsonr(s[a], s[b])
            corr_rows.append(dict(gene=g, comparison=desc, n=len(s),
                Spearman_rho=round(rho,3), Spearman_p=round(p,4),
                Pearson_r=round(r,3), Pearson_p=round(p2,4)))
            print(f"  [{g}] {desc}: rho={rho:+.3f} (p={p:.3g})  r={r:+.3f} (p={p2:.3g}) n={len(s)}")
pd.DataFrame(corr_rows).to_csv(os.path.join(OUT, "dual_axis_correlations.csv"), index=False)

# ---------------- figures ----------------
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rng = np.random.default_rng(0)

def umap_fig(adata, color, title, fname):
    n = min(adata.n_obs, 30000)
    vis = adata[adata.obs_names.values[rng.choice(adata.n_obs, n, replace=False)]].copy()
    sc.pp.neighbors(vis, n_neighbors=15, n_pcs=30); sc.tl.umap(vis)
    plt.figure(figsize=(7,6))
    sc.pl.umap(vis, color=color, show=False, title=title, ax=plt.gca())
    plt.tight_layout(); plt.savefig(os.path.join(OUT, fname), dpi=150); plt.close()

umap_fig(imm_ad, "lineage", "CD45+ immune lineages", "dual_umap_immune.png")
umap_fig(tn_ad,  "tumclass", "CD45n tumor/stroma classification", "dual_umap_tumor.png")
for g in GENES:
    if (imm_vec[g] is not None) or (tn_vec[g] is not None):
        # combined light umap of both compartments for gene expression
        comb2 = sc.concat([imm_ad, tn_ad], join="inner", index_unique="-")
        gv = gvec(comb2, g)
        comb2.obs[g] = gv
        n = min(comb2.n_obs, 40000)
        vis = comb2[comb2.obs_names.values[rng.choice(comb2.n_obs, n, replace=False)]].copy()
        sc.pp.neighbors(vis, n_neighbors=15, n_pcs=30); sc.tl.umap(vis)
        plt.figure(figsize=(7,6))
        sc.pl.umap(vis, color=g, show=False, title=f"{g} expression (log1p)", ax=plt.gca(),
                   vmin=0, vmax=3, cmap="viridis")
        plt.tight_layout(); plt.savefig(os.path.join(OUT, f"dual_exp_umap_{g}.png"), dpi=150); plt.close()

# KEY scatter: tumor DEPTOR vs myeloid DEPTOR
def annotate_scatter(ax, sub, x, y, df):
    for _, rr in df.iterrows():
        if pd.notna(rr[x]) and pd.notna(rr[y]):
            ax.annotate(rr.patient, (rr[x], rr[y]), fontsize=7, xytext=(3,3),
                        textcoords="offset points")

if "DEPTOR_malignant_mean" in df and "DEPTOR_myeloid_mean" in df:
    s = df[["DEPTOR_malignant_mean","DEPTOR_myeloid_mean"]].dropna()
    rho, p = spearmanr(s.DEPTOR_malignant_mean, s.DEPTOR_myeloid_mean)
    r, p2 = pearsonr(s.DEPTOR_malignant_mean, s.DEPTOR_myeloid_mean)
    plt.figure(figsize=(6,5))
    plt.scatter(s.DEPTOR_malignant_mean, s.DEPTOR_myeloid_mean, c="#2ca02c", s=60, edgecolor="k")
    annotate_scatter(plt.gca(), s, "DEPTOR_malignant_mean", "DEPTOR_myeloid_mean", df)
    plt.xlabel("DEPTOR mean in malignant (tumor) cells")
    plt.ylabel("DEPTOR mean in myeloid (immune) cells")
    plt.title(f"Tumor vs myeloid DEPTOR\nSpearman rho={rho:+.2f} (p={p:.2g}); Pearson r={r:+.2f} (p={p2:.2g})")
    plt.tight_layout(); plt.savefig(os.path.join(OUT,"dual_tumor_vs_myeloid_DEPTOR.png"), dpi=150); plt.close()

# scatter: tumor DEPTOR vs NK fraction
s2 = df[["DEPTOR_malignant_mean","NK_fraction_of_immune"]].dropna()
if len(s2) >= 4:
    rho, p = spearmanr(s2.DEPTOR_malignant_mean, s2.NK_fraction_of_immune)
    plt.figure(figsize=(6,5))
    plt.scatter(s2.DEPTOR_malignant_mean, s2.NK_fraction_of_immune, c="#9467bd", s=60, edgecolor="k")
    annotate_scatter(plt.gca(), s2, "DEPTOR_malignant_mean", "NK_fraction_of_immune", df)
    plt.xlabel("DEPTOR mean in malignant (tumor) cells")
    plt.ylabel("NK fraction of immune")
    plt.title(f"Tumor DEPTOR vs NK fraction\nSpearman rho={rho:+.2f} (p={p:.2g})")
    plt.tight_layout(); plt.savefig(os.path.join(OUT,"dual_tumorDEPTOR_vs_NK.png"), dpi=150); plt.close()

# scatter: myeloid DEPTOR vs NK fraction (replicate prior finding)
s3 = df[["DEPTOR_myeloid_mean","NK_fraction_of_immune"]].dropna()
if len(s3) >= 4:
    rho, p = spearmanr(s3.DEPTOR_myeloid_mean, s3.NK_fraction_of_immune)
    plt.figure(figsize=(6,5))
    plt.scatter(s3.DEPTOR_myeloid_mean, s3.NK_fraction_of_immune, c="#1f77b4", s=60, edgecolor="k")
    annotate_scatter(plt.gca(), s3, "DEPTOR_myeloid_mean", "NK_fraction_of_immune", df)
    plt.xlabel("DEPTOR mean in myeloid (immune) cells")
    plt.ylabel("NK fraction of immune")
    plt.title(f"Myeloid DEPTOR vs NK fraction\nSpearman rho={rho:+.2f} (p={p:.2g})")
    plt.tight_layout(); plt.savefig(os.path.join(OUT,"dual_myeloidDEPTOR_vs_NK.png"), dpi=150); plt.close()

print("\nSaved to", OUT); print("DONE.")
