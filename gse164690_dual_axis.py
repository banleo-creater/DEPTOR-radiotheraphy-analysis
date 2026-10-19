#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GSE164690 (Kürten et al. Nat Commun 2021) — DUAL-AXIS pilot.
Combines tumor-infiltrating immune (CD45+) and tumor+stroma (CD45n) 10x samples.
Replicates cell types de novo, then asks:
  (A) DEPTOR in malignant (tumor) cells  vs  DEPTOR in myeloid (immune) cells
      across patients -> are they anti-correlated? (the "tumor vs immune" dual axis)
  (B) DEPTOR in malignant / myeloid  vs  NK abundance in CD45+.
EPYC kept for completeness (earlier found near-absent in myeloid).
"""
import os, re, glob, warnings
import numpy as np, pandas as pd
import scanpy as sc
warnings.filterwarnings("ignore"); sc.settings.verbosity = 0

BASE = "WorkBuddy/2026-10-17-11-15-46"
RAW  = os.path.join(BASE, "data", "raw")
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

def sample_meta(name):
    m = re.search(r"(HN\d+)_(CD45[pn]|PBL)", name)
    patient = m.group(1) if m else name
    comp = {"CD45p":"Immune","CD45n":"TumorStroma","PBL":"Blood"}[m.group(2)] if m else "Other"
    return patient, comp

print("Reading CD45+ and CD45n sample directories ...")
adatas = []
for d in sorted(glob.glob(os.path.join(RAW, "*"))):
    if not os.path.isdir(d): continue
    sample = os.path.basename(d)
    if "CD45p" not in sample and "CD45n" not in sample: continue
    try:
        ad = sc.read_10x_mtx(d, var_names="gene_symbols", cache=False)
    except Exception as e:
        print("  skip", sample, e); continue
    ad.var_names_make_unique()
    pat, comp = sample_meta(sample)
    ad.obs["sample"] = sample; ad.obs["patient"] = pat; ad.obs["compartment"] = comp
    adatas.append(ad); print(f"  loaded {sample}: {ad.n_obs} cells")
if not adatas:
    raise SystemExit("No CD45p/CD45n samples found.")

adata = sc.concat(adatas, join="inner", index_unique="-")
print(f"\nCombined: {adata.n_obs} cells x {adata.n_vars} genes")
del adatas
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.normalize_total(adata, target_sum=1e4); sc.pp.log1p(adata)

print("Target genes:", {g: (g in adata.var_names) for g in GENES})

X = adata.X
vidx = {str(v): i for i, v in enumerate(adata.var_names)}
def score(mk):
    idxs = [vidx[m] for m in mk if m in vidx]
    if not idxs: return np.zeros(adata.n_obs)
    sub = X[:, idxs].mean(axis=1)
    if hasattr(sub, "todense"): sub = sub.todense()
    return np.asarray(sub).ravel()

# ---- immune lineage (CD45p) ----
imm = adata.obs["compartment"].values == "Immune"
smat = np.vstack([score(mk) for mk in IMMUNE_MARKERS.values()]).T
lin = np.array(list(IMMUNE_MARKERS.keys()))[smat.argmax(axis=1)]
lin[smat.max(axis=1) < 0.05] = "Other"
adata.obs["lineage"] = lin
mye = adata.obs["lineage"].values == "Myeloid"
sm, sd = score(MAC_MARKERS), score(DC_MARKERS)
adata.obs.loc[adata.obs.index[mye], "lineage"] = np.where(sm[mye] >= sd[mye], "Macrophage", "DC")

# ---- tumor/stroma classification (CD45n) ----
tn = adata.obs["compartment"].values == "TumorStroma"
epi_s, stro_s, endo_s = score(TUMOR_MARKERS["Malignant"]), score(TUMOR_MARKERS["Stroma"]), score(TUMOR_MARKERS["Endothelial"])
imm_leak = score(IMMUNE_LEAK)
tclass = np.full(adata.n_obs, "Other", dtype=object)
# priority: immune leak -> drop to Other(immune contam); else malignant vs stroma vs endo by max
idx_tn = np.where(tn)[0]
for i in idx_tn:
    if imm_leak[i] > 0.5:
        tclass[i] = "ImmuneContam"; continue
    a,b,c = epi_s[i], stro_s[i], endo_s[i]
    if a >= b and a >= c and a > 0.3: tclass[i] = "Malignant"
    elif b >= c and b > 0.3: tclass[i] = "Stroma"
    elif c > 0.3: tclass[i] = "Endothelial"
    else: tclass[i] = "Other"
adata.obs["tumclass"] = tclass

print("\nImmune (CD45+) composition:"); print(adata.obs.loc[adata.obs.compartment=="Immune","lineage"].value_counts().to_string())
print("\nTumor/Stroma (CD45n) composition:"); print(adata.obs.loc[adata.obs.compartment=="TumorStroma","tumclass"].value_counts().to_string())

# ---- dense gene vectors ----
gvec = {}
for g in GENES:
    gi = vidx.get(g)
    col = adata.X[:, gi]
    if hasattr(col, "todense"): col = col.todense()
    gvec[g] = np.asarray(col).ravel() if gi is not None else None

names = adata.obs_names.values
linv = adata.obs["lineage"].values
tclv = adata.obs["tumclass"].values
compv = adata.obs["compartment"].values
mye_classes = np.isin(linv, ["Myeloid","Macrophage","DC"])

# ---- per-patient aggregates ----
rows = []
for pat, sub in adata.obs.groupby("patient"):
    mask = np.isin(names, sub.index.values)
    n_imm = int((compv[mask]=="Immune").sum())
    n_nk = int(((compv[mask]=="Immune") & (linv[mask]=="NK")).sum())
    n_mye= int(((compv[mask]=="Immune") & mye_classes[mask]).sum())
    n_mal= int(((compv[mask]=="TumorStroma") & (tclv[mask]=="Malignant")).sum())
    n_str= int(((compv[mask]=="TumorStroma") & (tclv[mask]=="Stroma")).sum())
    n_end= int(((compv[mask]=="TumorStroma") & (tclv[mask]=="Endothelial")).sum())
    rec = dict(patient=pat, n_immune=n_imm, n_NK=n_nk, n_myeloid=n_mye,
               n_malignant=n_mal, n_stroma=n_str, n_endothelial=n_end,
               NK_fraction_of_immune=(n_nk/n_imm if n_imm else np.nan))
    mal_mask = mask & (compv=="TumorStroma") & (tclv=="Malignant")
    str_mask = mask & (compv=="TumorStroma") & (tclv=="Stroma")
    mye_mask = mask & (compv=="Immune") & mye_classes
    for g in GENES:
        if gvec[g] is not None:
            rec[f"{g}_malignant_mean"] = float(gvec[g][mal_mask].mean())
            rec[f"{g}_malignant_detect"] = float((gvec[g][mal_mask]>0).mean())
            rec[f"{g}_stroma_mean"] = float(gvec[g][str_mask].mean())
            rec[f"{g}_stroma_detect"] = float((gvec[g][str_mask]>0).mean())
            rec[f"{g}_myeloid_mean"] = float(gvec[g][mye_mask].mean())
            rec[f"{g}_myeloid_detect"] = float((gvec[g][mye_mask]>0).mean())
        else:
            for s in ["malignant","stroma","myeloid"]:
                rec[f"{g}_{s}_mean"]=rec[f"{g}_{s}_detect"]=np.nan
    rows.append(rec)

df = pd.DataFrame(rows).sort_values("patient").reset_index(drop=True)
df.to_csv(os.path.join(OUT,"dual_axis_summary.csv"), index=False)
print("\n=== Patient-level dual-axis summary ===")
print(df.to_string(index=False))

# ---- correlations: the dual axis ----
from scipy.stats import spearmanr, pearsonr
print("\n=== Dual-axis correlations (patient level) ===")
corr_rows = []
for g in GENES:
    pairs = [
        (f"{g}_malignant_mean", f"{g}_myeloid_mean", "tumor DEPTOR vs myeloid DEPTOR"),
        (f"{g}_malignant_mean", "NK_fraction_of_immune", "tumor DEPTOR vs NK fraction"),
        (f"{g}_myeloid_mean",   "NK_fraction_of_immune", "myeloid DEPTOR vs NK fraction"),
        (f"{g}_malignant_mean", f"{g}_stroma_mean", "tumor DEPTOR vs stroma DEPTOR"),
    ]
    for a,b,desc in pairs:
        s = df[[a,b]].dropna()
        if len(s) >= 4:
            rho,p = spearmanr(s[a], s[b]); r,p2 = pearsonr(s[a], s[b])
            corr_rows.append(dict(gene=g, comparison=desc, n=len(s),
                Spearman_rho=round(rho,3), Spearman_p=round(p,4),
                Pearson_r=round(r,3), Pearson_p=round(p2,4)))
            print(f"  [{g}] {desc}: rho={rho:+.3f} (p={p:.3g})  r={r:+.3f} (p={p2:.3g}) n={len(s)}")
pd.DataFrame(corr_rows).to_csv(os.path.join(OUT,"dual_axis_correlations.csv"), index=False)

# ---- figures ----
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rng = np.random.default_rng(0)
n_vis = min(adata.n_obs, 30000)
vis = adata[names[rng.choice(adata.n_obs, n_vis, replace=False)]].copy()
sc.pp.neighbors(vis, n_neighbors=15, n_pcs=30); sc.tl.umap(vis)

plt.figure(figsize=(7,6))
sc.pl.umap(vis[vis.obs.compartment=="Immune"], color="lineage", show=False,
           title="CD45+ immune lineages", ax=plt.gca()); plt.tight_layout()
plt.savefig(os.path.join(OUT,"dual_umap_immune.png"), dpi=150); plt.close()

plt.figure(figsize=(7,6))
sc.pl.umap(vis[vis.obs.compartment=="TumorStroma"], color="tumclass", show=False,
           title="CD45n tumor/stroma classification", ax=plt.gca()); plt.tight_layout()
plt.savefig(os.path.join(OUT,"dual_umap_tumor.png"), dpi=150); plt.close()

for g in GENES:
    if gvec[g] is not None:
        plt.figure(figsize=(7,6))
        sc.pl.umap(vis, color=g, show=False, title=f"{g} expression (log1p)",
                   ax=plt.gca(), vmin=0, vmax=3, cmap="viridis")
        plt.tight_layout(); plt.savefig(os.path.join(OUT,f"dual_exp_umap_{g}.png"), dpi=150); plt.close()

# KEY scatter: tumor DEPTOR vs myeloid DEPTOR
if "DEPTOR_malignant_mean" in df and "DEPTOR_myeloid_mean" in df:
    s = df[["DEPTOR_malignant_mean","DEPTOR_myeloid_mean"]].dropna()
    rho,p = spearmanr(s.DEPTOR_malignant_mean, s.DEPTOR_myeloid_mean)
    plt.figure(figsize=(6,5))
    plt.scatter(s.DEPTOR_malignant_mean, s.DEPTOR_myeloid_mean, c="#2ca02c", s=60, edgecolor="k")
    for _,r in df.iterrows():
        if pd.notna(r.DEPTOR_malignant_mean) and pd.notna(r.DEPTOR_myeloid_mean):
            plt.annotate(r.patient, (r.DEPTOR_malignant_mean, r.DEPTOR_myeloid_mean),
                         fontsize=7, xytext=(3,3), textcoords="offset points")
    plt.xlabel("DEPTOR mean in malignant (tumor) cells"); plt.ylabel("DEPTOR mean in myeloid (immune) cells")
    plt.title(f"Tumor vs myeloid DEPTOR\nSpearman rho={rho:+.2f}, p={p:.2g}")
    plt.tight_layout(); plt.savefig(os.path.join(OUT,"dual_tumor_vs_myeloid_DEPTOR.png"), dpi=150); plt.close()

# scatter: tumor DEPTOR vs NK fraction
s = df[["DEPTOR_malignant_mean","NK_fraction_of_immune"]].dropna()
if len(s)>=4:
    rho,p = spearmanr(s.DEPTOR_malignant_mean, s.NK_fraction_of_immune)
    plt.figure(figsize=(6,5))
    plt.scatter(s.DEPTOR_malignant_mean, s.NK_fraction_of_immune, c="#9467bd", s=60, edgecolor="k")
    for _,r in df.iterrows():
        if pd.notna(r.DEPTOR_malignant_mean) and pd.notna(r.NK_fraction_of_immune):
            plt.annotate(r.patient, (r.DEPTOR_malignant_mean, r.NK_fraction_of_immune),
                         fontsize=7, xytext=(3,3), textcoords="offset points")
    plt.xlabel("DEPTOR mean in malignant (tumor) cells"); plt.ylabel("NK fraction of immune")
    plt.title(f"Tumor DEPTOR vs NK fraction\nSpearman rho={rho:+.2f}, p={p:.2g}")
    plt.tight_layout(); plt.savefig(os.path.join(OUT,"dual_tumorDEPTOR_vs_NK.png"), dpi=150); plt.close()

print("\nSaved to", OUT); print("DONE.")
