#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GSE164690 (Kürten et al. Nat Commun 2021) feasibility pilot  —  EPYC + DEPTOR
Reads the selectively-downloaded CD45+ (tumor immune) 10x samples from
data/raw/<sample>/ and tests, for BOTH target genes (EPYC and DEPTOR), whether
their expression correlates with NK-cell abundance and myeloid (TAM/MDSC)
composition in the HNSCC TME.

Cell types are recovered *de novo* from canonical markers (no external labels).
"""
import os, re, glob, warnings
import numpy as np
import pandas as pd
import scanpy as sc

warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

BASE = "WorkBuddy/2026-10-17-11-15-46"
RAW  = os.path.join(BASE, "data", "raw")
OUT  = os.path.join(BASE, "data", "results")
os.makedirs(OUT, exist_ok=True)

GENES = ["EPYC", "DEPTOR"]   # both target genes

MARKERS = {
    "Tcell":      ["CD3D","CD3E","CD8A","CD8B","CD4","IL7R","CD2"],
    "NK":         ["NCAM1","NKG7","KLRF1","KLRD1","FCGR3A","GNLY","TBX21",
                   "PRF1","GZMB","KLRK1","XCL1","XCL2","FCGR3B"],
    "Bcell":      ["CD79A","CD79B","MS4A1","JCHAIN","MZB1","CD19"],
    "Myeloid":    ["LYZ","CD68","CD163","CSF1R","FCGR1A","ITGAM","VSIG4",
                   "MRC1","HLA-DRA","CD14","SPI1","AIF1","LST1"],
}
MAC_MARKERS = ["CD163","VSIG4","MRC1","CD14","FCGR1A","SPI1","APOE","FABP4","C1QA","C1QB"]
DC_MARKERS  = ["FCER1A","IRF8","CLEC9A","CD1C","LILRA4","CCR7","BATF3"]

def sample_meta(name):
    m = re.search(r"(HN\d+)_(CD45[pn]|PBL)", name)
    patient = m.group(1) if m else name
    comp = {"CD45p":"Immune","CD45n":"NonImmune","PBL":"Blood"}[m.group(2)] if m else "Other"
    return patient, comp

print("Reading CD45+ (immune) sample directories ...")
adatas = []
for d in sorted(glob.glob(os.path.join(RAW, "*"))):
    if not os.path.isdir(d):
        continue
    sample = os.path.basename(d)
    if "CD45p" not in sample:        # only tumor immune
        continue
    try:
        ad = sc.read_10x_mtx(d, var_names="gene_symbols", cache=False)
    except Exception as e:
        print("  skip", sample, e); continue
    ad.var_names_make_unique()
    pat, comp = sample_meta(sample)
    ad.obs["sample"] = sample
    ad.obs["patient"] = pat
    ad.obs["compartment"] = comp
    adatas.append(ad)
    print(f"  loaded {sample}: {ad.n_obs} cells")

if not adatas:
    raise SystemExit("No CD45p samples found — download may still be running.")

adata = sc.concat(adatas, join="inner", index_unique="-")
print(f"\nCombined: {adata.n_obs} cells x {adata.n_vars} genes")
del adatas

# QC + normalize
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

print("\nTarget gene presence:")
present = {}
for g in GENES:
    if g in adata.var_names:
        present[g] = g
    else:
        hit = [v for v in adata.var_names if str(v).upper() == g]
        present[g] = hit[0] if hit else None
    print(f"  {g}: {'FOUND ' + present[g] if present[g] else 'ABSENT'}")

# marker scoring
X = adata.X
vidx = {str(v): i for i, v in enumerate(adata.var_names)}
def score(mk):
    idxs = [vidx[m] for m in mk if m in vidx]
    if not idxs:
        return np.zeros(adata.n_obs)
    sub = X[:, idxs].mean(axis=1)
    if hasattr(sub, "todense"):
        sub = sub.todense()
    return np.asarray(sub).ravel()

smat = np.vstack([score(mk) for mk in MARKERS.values()]).T
lineage = np.array(list(MARKERS.keys()))[smat.argmax(axis=1)]
lineage[smat.max(axis=1) < 0.05] = "Other"
adata.obs["lineage"] = lineage
mye = adata.obs["lineage"] == "Myeloid"
if mye.sum() > 0:
    sm, sd = score(MAC_MARKERS), score(DC_MARKERS)
    adata.obs.loc[adata.obs.index[mye], "lineage"] = np.where(sm[mye] >= sd[mye], "Macrophage", "DC")

print("\nCell-type composition (tumor immune / CD45+):")
print(adata.obs["lineage"].value_counts().to_string())

# per-patient aggregates for EACH gene
# extract dense per-cell vectors for target genes once (adata.X is sparse)
gvec = {}
for g in GENES:
    gi = vidx.get(g)
    if gi is not None and present[g]:
        col = adata.X[:, gi]
        if hasattr(col, "todense"):
            col = col.todense()
        gvec[g] = np.asarray(col).ravel()
    else:
        gvec[g] = None

lin = adata.obs["lineage"].values
names = adata.obs_names.values
mye_classes = np.isin(lin, ["Myeloid","Macrophage","DC"])

rows = []
all_res = {}
for pat, sub in adata.obs.groupby("patient"):
    n_imm = sub.shape[0]
    n_nk = (sub["lineage"]=="NK").sum()
    n_t  = (sub["lineage"]=="Tcell").sum()
    n_b  = (sub["lineage"]=="Bcell").sum()
    n_mye= (sub["lineage"].isin(["Myeloid","Macrophage","DC"])).sum()
    n_mac= (sub["lineage"]=="Macrophage").sum()
    n_dc = (sub["lineage"]=="DC").sum()
    mask = np.isin(names, sub.index.values)
    rec = dict(patient=pat, n_immune=int(n_imm), n_NK=int(n_nk), n_T=int(n_t),
        n_B=int(n_b), n_myeloid=int(n_mye), n_mac=int(n_mac), n_dc=int(n_dc),
        NK_fraction_of_immune = n_nk/n_imm if n_imm else np.nan,
        NK_fraction_of_lymph = n_nk/(n_nk+n_t+n_b) if (n_nk+n_t+n_b) else np.nan)
    for g in GENES:
        if gvec[g] is not None:
            gm = float(gvec[g][mask & mye_classes].mean())
            gd = float((gvec[g][mask & mye_classes] > 0).mean())
            imd = float((gvec[g][mask] > 0).mean())
            rec[f"{g}_myeloid_mean"] = gm
            rec[f"{g}_myeloid_detect"] = gd
            rec[f"{g}_immune_detect"] = imd
        else:
            rec[f"{g}_myeloid_mean"] = rec[f"{g}_myeloid_detect"] = rec[f"{g}_immune_detect"] = np.nan
    rows.append(rec)

df = pd.DataFrame(rows).sort_values("patient").reset_index(drop=True)
df.to_csv(os.path.join(OUT,"epyc_deptor_pilot_summary.csv"), index=False)
print("\n=== Patient-level summary ===")
print(df.to_string(index=False))

# correlations for each gene
from scipy.stats import spearmanr, pearsonr
print("\n=== Correlations (patient level: NK fraction vs gene) ===")
for g in GENES:
    res = all_res[g] = {}
    for col in [f"{g}_myeloid_mean", f"{g}_myeloid_detect", f"{g}_immune_detect"]:
        s = df[[col,"NK_fraction_of_immune"]].dropna()
        if len(s) >= 3:
            rho,p = spearmanr(s[col], s["NK_fraction_of_immune"])
            r,p2 = pearsonr(s[col], s["NK_fraction_of_immune"])
            res[col] = (rho,p,r,p2)
            print(f"  [{g}] NK_fraction vs {col}: Spearman rho={rho:+.3f} (p={p:.3g})  "
                  f"Pearson r={r:+.3f} (p={p2:.3g})  n={len(s)}")
        else:
            print(f"  [{g}] NK_fraction vs {col}: insufficient data (n={len(s)})")

# visualizations
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
rng = np.random.default_rng(0)
n_vis = min(adata.n_obs, 25000)
vis = adata[adata.obs.index.values[rng.choice(adata.n_obs, n_vis, replace=False)]].copy()
sc.pp.neighbors(vis, n_neighbors=15, n_pcs=30)
sc.tl.umap(vis)

plt.figure(figsize=(7,6))
sc.pl.umap(vis, color="lineage", show=False, title="GSE164690 HNSCC — tumor immune (CD45+) lineages",
           save=None, ax=plt.gca())
plt.tight_layout(); plt.savefig(os.path.join(OUT,"epyc_deptor_umap_celltype.png"), dpi=150); plt.close()

# UMAP expression per gene
for g in GENES:
    if present[g]:
        plt.figure(figsize=(7,6))
        sc.pl.umap(vis, color=g, show=False, title=f"{g} expression (log1p)",
                   save=None, ax=plt.gca(), vmin=0, vmax=3, cmap="viridis")
        plt.tight_layout(); plt.savefig(os.path.join(OUT,f"exp_umap_{g}.png"), dpi=150); plt.close()

# scatter per gene (NK fraction vs myeloid mean)
for g in GENES:
    col = f"{g}_myeloid_mean"
    if g in all_res and all_res[g].get(col) is not None:
        rho,p = all_res[g][col][:2]
        plt.figure(figsize=(6,5))
        plt.scatter(df[col], df["NK_fraction_of_immune"], c="#C44E52", s=60, edgecolor="k")
        for _,r in df.iterrows():
            if pd.notna(r[col]):
                plt.annotate(r["patient"], (r[col], r["NK_fraction_of_immune"]),
                             fontsize=7, xytext=(3,3), textcoords="offset points")
        plt.xlabel(f"{g} mean expression in myeloid (TAM/MDSC)")
        plt.ylabel("NK fraction of immune cells")
        plt.title(f"NK fraction vs {g}(myeloid)\nSpearman rho={rho:+.2f}, p={p:.2g}")
        plt.tight_layout(); plt.savefig(os.path.join(OUT,f"nk_corr_{g}.png"), dpi=150); plt.close()

# combined bar comparison of detection rates + correlation summary
fig, axes = plt.subplots(1, 2, figsize=(12,5))
for ax,(g,lab) in zip(axes,[("EPYC","EPYC"),("DEPTOR","DEPTOR")]):
    dm = df[f"{g}_myeloid_detect"].dropna()*100
    dd = df[f"{g}_immune_detect"].dropna()*100
    ax.bar(["myeloid\n(TAM/MDSC)","immune\n(all CD45+)"],[dm.mean(),dd.mean()],
           color=["#4C72B0","#DD8452"], edgecolor="k")
    ax.set_ylabel("% cells expressing (detect rate)")
    ax.set_title(f"{lab} detection rate\n(myeloid mean det={dm.mean():.1f}%)")
    for i,v in enumerate([dm.mean(),dd.mean()]):
        ax.text(i,v+1,f"{v:.1f}%",ha="center",fontsize=9)
fig.suptitle("EPYC vs DEPTOR detection in HNSCC tumor immune (GSE164690)")
plt.tight_layout(); plt.savefig(os.path.join(OUT,"epyc_deptor_compare.png"), dpi=150); plt.close()

print("\nSaved to", OUT)
print("DONE.")
