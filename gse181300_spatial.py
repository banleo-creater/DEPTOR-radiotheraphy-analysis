#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GSE181300 (Cheng et al. 2022, 10x Visium, 8 HNSCC sections, 2 patients,
invasive-front vs inner-site annotated) — SPATIAL VALIDATION of the DEPTOR
"tumor / immune dual-axis -> NK exclusion" hypothesis from GSE164690 scRNA.

Key data property (diagnosed): DEPTOR is detected in only ~5% of spots
(per-sample dropout 0.91–1.00) and EPYC in ~0.1%. Spot-level continuous
Moran/Spearman on these sparse vectors is a zero-inflation artifact, so it is
NOT used as the primary evidence.

Strategy (what is reliable at spot resolution):
  1. Standard Visium preprocessing (normalize + log1p).
  2. De novo cell-type program scoring per spot (Tumor/Tcell/Bcell/Mac/cDC/NK/
     Stroma/Endo) from canonical markers -> dominant label + softmax proportion.
  3. WHERE IS DEPTOR IN SPACE? Test whether DEPTOR-present spots map onto the
     malignant-epithelium compartment (consistent with scRNA: malignant DEPTOR
     is the high-expressing axis).
  4. Cell-type neighbourhood enrichment (kNN + label permutation) per sample,
     then aggregated -> which cell types co-localise / exclude in the niche
     (e.g. Tumor<->NK, Tumor<->Mac, Mac<->NK).
  5. DEPTOR-present vs DEPTOR-absent TUMOR spots: are neighbours NK-poor /
     TAM-rich? (direct spatial test, reported with power caveat).
  6. Invasive-front vs inner-site immune composition stratification.

Efficiency: kNN kept as an index array (no dense n x n W); neighbour summaries
use V[idx].mean(axis=1).
"""
import os, json, warnings
import numpy as np, pandas as pd
import scanpy as sc
from scipy.spatial import cKDTree
from scipy.stats import spearmanr, mannwhitneyu, norm as normdist
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore"); sc.settings.verbosity = 0

BASE = "WorkBuddy/2026-10-17-11-15-46"
RAW  = os.path.join(BASE, "data", "gse181300_raw")
OUT  = os.path.join(BASE, "data", "results_spatial"); os.makedirs(OUT, exist_ok=True)

SAMPLES = [
    ("HNSCC201125T04","upn1","invasive_front"),
    ("HNSCC201125T05","upn1","invasive_front"),
    ("HNSCC201125T07","upn1","inner_site"),
    ("HNSCC201125T10","upn1","inner_site"),
    ("P210325T1","upn2","invasive_front"),
    ("P210325T3","upn2","invasive_front"),
    ("P210325T5","upn2","inner_site"),
    ("P210325T6","upn2","inner_site"),
]

MARKERS = {
    "Tumor": ["KRT5","KRT6A","KRT8","KRT14","EPCAM","TP63","KRT19"],
    "Tcell": ["CD3D","CD3E","CD8A","CD4","IL7R","CD2"],
    "Bcell": ["CD79A","CD79B","MS4A1","JCHAIN","CD19"],
    "Mac":   ["CD68","CD163","MRC1","LYZ","CSF1R","VSIG4","AIF1"],
    "cDC":   ["CD1C","FCER1A","IRF4","HLA-DRA","CLEC9A","CCR7"],
    "NK":    ["NCAM1","NKG7","KLRF1","KLRD1","GNLY","XCL1","XCL2","KLRK1"],
    "Stroma":["COL1A1","COL1A2","ACTA2","PDGFRA","FAP","DCN"],
    "Endo":  ["PECAM1","VWF","CLDN5","CDH5"],
}
TYPES = list(MARKERS.keys())
K_T = len(TYPES)

# ---------- helpers ----------
def score(adata, mk):
    vidx = {str(v): i for i, v in enumerate(adata.var_names)}
    idxs = [vidx[m] for m in mk if m in vidx]
    if not idxs:
        return np.zeros(adata.n_obs)
    sub = adata.X[:, idxs]
    if hasattr(sub, "todense"): sub = sub.todense()
    return np.asarray(sub).mean(axis=1).ravel()

def load_sample(name):
    d = os.path.join(RAW, name)
    h5 = os.path.join(d, "filtered_feature_bc_matrix.h5")
    ad = sc.read_10x_h5(h5)
    ad.var_names_make_unique()
    sc.pp.filter_genes(ad, min_cells=1)
    sc.pp.normalize_total(ad, target_sum=1e4); sc.pp.log1p(ad)
    posfiles = [os.path.join(d,"spatial",f) for f in
                ["tissue_positions.csv","tissue_positions_list.csv"]]
    posfile = next((p for p in posfiles if os.path.exists(p)), None)
    if posfile is None:
        raise SystemExit(f"no tissue_positions for {name}")
    pos = pd.read_csv(posfile)
    if "barcode" not in pos.columns:
        pos.columns = ["barcode","in_tissue","array_row","array_col",
                       "pxl_row_in_fullres","pxl_col_in_fullres"]
    pos = pos.set_index("barcode")
    common = [b for b in ad.obs_names if b in pos.index]
    ad = ad[common].copy()
    pr = pos.loc[common, "pxl_row_in_fullres"].values.astype(float)
    pc = pos.loc[common, "pxl_col_in_fullres"].values.astype(float)
    ad.obsm["spatial"] = np.column_stack([pc, pr])
    ad.obs["in_tissue"] = pos.loc[common, "in_tissue"].values
    sf = os.path.join(d,"spatial","scalefactors_json.json")
    if os.path.exists(sf):
        with open(sf) as f: ad.uns["scalefactors"] = json.load(f)
    return ad

def build_knn(coords, k=6):
    tree = cKDTree(coords)
    dist, idx = tree.query(coords, k=k+1)
    return idx[:, 1:].astype(np.int64)

def softmax(x, t=1.0):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x / t); return e / e.sum(axis=1, keepdims=True)

def onehot(labels, K):
    n = len(labels)
    Oh = np.zeros((n, K)); Oh[np.arange(n), labels] = 1
    return Oh

def neighbour_avg(nbr_idx, V):
    return V[nbr_idx].mean(axis=1)

def enrichment_perm(nbr_idx, labels, nperm=999, seed=0):
    K = K_T
    nc = neighbour_avg(nbr_idx, onehot(labels, K))
    obs = np.zeros((K, K))
    for A in range(K):
        m = labels == A
        if m.sum(): obs[A] = nc[m].mean(axis=0)
    rng = np.random.default_rng(seed)
    nm = np.zeros((K, K)); ns = np.zeros((K, K))
    lp = labels.copy()
    for _ in range(nperm):
        rng.shuffle(lp)
        ncp = neighbour_avg(nbr_idx, onehot(lp, K))
        o2 = np.zeros((K, K))
        for A in range(K):
            mm = lp == A
            if mm.sum(): o2[A] = ncp[mm].mean(axis=0)
        nm += o2; ns += o2**2
    nm /= nperm
    sd = np.sqrt(np.maximum(ns/nperm - nm**2, 1e-12))
    z = (obs - nm) / sd
    p = 2 * (1 - normdist.cdf(np.abs(z)))
    return obs, z, p

# ---------- main loop ----------
print("Loading Visium sections ...")
records = []; meta = []
for name, pat, region in SAMPLES:
    try:
        ad = load_sample(name)
    except Exception as e:
        print(f"  SKIP {name}: {e}"); continue
    it = ad.obs["in_tissue"].values.astype(float) > 0
    ad_it = ad[it].copy()
    coords = ad_it.obsm["spatial"]
    nbr_idx = build_knn(coords, k=6)
    S = np.column_stack([score(ad_it, MARKERS[t]) for t in TYPES])
    labels = np.argmax(S, axis=1)
    prop = softmax(S, t=0.5)
    vidx = {str(v): i for i, v in enumerate(ad_it.var_names)}
    dv = ad_it.X[:, vidx["DEPTOR"]]
    dv = np.asarray(dv.todense()).ravel() if hasattr(dv,"todense") else dv.ravel()
    dropout = float((dv == 0).mean())
    # neighbour composition (continuous proportions)
    nc_prop = neighbour_avg(nbr_idx, prop)
    iNK, iMac, iTum = TYPES.index("NK"), TYPES.index("Mac"), TYPES.index("Tumor")
    is_tumor = labels == iTum
    dpTile = (dv > 0)
    tp_p = is_tumor & dpTile
    tp_a = is_tumor & (~dpTile)
    def _m(v): return float(np.mean(v)) if v.size else np.nan
    rec = dict(sample=name, patient=pat, region=region, n_spots=int(it.sum()),
               DEPTOR_dropout=dropout, n_DEPTORpresent=int(dpTile.sum()),
               Tumor_prop=float(prop[:,iTum].mean()),
               NK_prop=float(prop[:,iNK].mean()),
               Mac_prop=float(prop[:,iMac].mean()),
               frac_DEPTORpresent_isTumor=float((labels[dpTile]==iTum).mean()) if dpTile.sum() else np.nan,
               frac_all_isTumor=float((labels==iTum).mean()),
               NK_neigh_present=_m(nc_prop[tp_p, iNK]), NK_neigh_absent=_m(nc_prop[tp_a, iNK]),
               Mac_neigh_present=_m(nc_prop[tp_p, iMac]), Mac_neigh_absent=_m(nc_prop[tp_a, iMac]))
    records.append(rec)
    meta.append(dict(ad=ad, ad_it=ad_it, name=name, pat=pat, region=region,
                     coords=coords, nbr_idx=nbr_idx, labels=labels, prop=prop, dv=dv))
    print(f"  {name:16s} {pat} {region:14s} spots={int(it.sum()):5d}  "
          f"DEPTOR_dropout={dropout:.2f}  n_DEPTOR+={int(dpTile.sum()):3d}  "
          f"NK_prop={rec['NK_prop']:.3f}  Mac_prop={rec['Mac_prop']:.3f}")

res = pd.DataFrame(records)
res.to_csv(os.path.join(OUT, "spatial_summary.csv"), index=False)
print("\n=== Per-section summary ===")
print(res[["sample","region","n_spots","DEPTOR_dropout","n_DEPTORpresent",
           "Tumor_prop","NK_prop","Mac_prop","frac_DEPTORpresent_isTumor","frac_all_isTumor",
           "NK_neigh_present","NK_neigh_absent","Mac_neigh_present","Mac_neigh_absent"]].to_string(index=False))

# ---- (3) does DEPTOR localise to malignant epithelium in space? ----
print("\n=== (3) Spatial localisation of DEPTOR ===")
dp_all = np.concatenate([m["dv"] > 0 for m in meta])
lab_all = np.concatenate([m["labels"] for m in meta])
reg_all = np.concatenate([np.full(len(m["labels"]), m["region"], dtype=object) for m in meta])
n_pres = int(dp_all.sum())
if n_pres:
    f_pres = float((lab_all[dp_all] == TYPES.index("Tumor")).mean())
    f_any  = float((lab_all == TYPES.index("Tumor")).mean())
    print(f"  DEPTOR-present spots pooled: n={n_pres}")
    print(f"    fraction labelled Tumor = {f_pres:.3f}")
    print(f"    same fraction among ALL spots = {f_any:.3f}")
    print(f"    -> DEPTOR is {'ENRICHED in' if f_pres>f_any else 'depleted of'} malignant-epithelium spots "
          f"(ratio {f_pres/max(f_any,1e-9):.2f}x)")

# ---- (5) DEPTOR-present vs absent tumor spots: neighbour immunity ----
print("\n=== (5) DEPTOR-present vs absent TUMOR spots: neighbour immunity ===")
P_nk, A_nk, P_mac, A_mac = [], [], [], []
for m in meta:
    is_t = m["labels"] == TYPES.index("Tumor")
    dpos = m["dv"] > 0
    tp_p = is_t & dpos; tp_a = is_t & (~dpos)
    nc = neighbour_avg(m["nbr_idx"], m["prop"])
    if tp_p.sum() and tp_a.sum():
        P_nk.append(nc[tp_p, TYPES.index("NK")]); A_nk.append(nc[tp_a, TYPES.index("NK")])
        P_mac.append(nc[tp_p, TYPES.index("Mac")]); A_mac.append(nc[tp_a, TYPES.index("Mac")])
if P_nk:
    P_nk=np.concatenate(P_nk); A_nk=np.concatenate(A_nk)
    P_mac=np.concatenate(P_mac); A_mac=np.concatenate(A_mac)
    print(f"  DEPTOR-present tumor spots: n={len(P_nk)} neighbour NK={P_nk.mean():.5f} Mac={P_mac.mean():.5f}")
    print(f"  DEPTOR-absent  tumor spots: n={len(A_nk)} neighbour NK={A_nk.mean():.5f} Mac={A_mac.mean():.5f}")
    _,p_k = mannwhitneyu(P_nk, A_nk, alternative="less")
    _,p_m = mannwhitneyu(P_mac, A_mac, alternative="greater")
    print(f"  -> neighbour NK lower when DEPTOR present: MWU one-sided p={p_k:.3g}")
    print(f"  -> neighbour Mac higher when DEPTOR present: MWU one-sided p={p_m:.3g}")
    print("  NOTE: DEPTOR-present tumor spots are few — interpret as exploratory.")
else:
    print("  insufficient DEPTOR-present tumor spots for comparison")

# ---- (4) neighbourhood enrichment, per sample then aggregated ----
print("\n=== (4) Cell-type neighbourhood enrichment (per-sample, aggregated) ===")
key_pairs = [("Tumor","NK"),("Tumor","Mac"),("Mac","NK"),("NK","Tumor"),("Mac","Tumor"),
             ("NK","Mac"),("Tumor","Tcell"),("Tumor","Stroma"),("Mac","Tcell")]
z_sum = np.zeros((K_T, K_T)); sig_cnt = np.zeros((K_T, K_T)); n_eff = 0
for m in meta:
    _, z_i, p_i = enrichment_perm(m["nbr_idx"], m["labels"], nperm=999, seed=1)
    z_sum += z_i; n_eff += 1
    sig_cnt += (p_i < 0.05).astype(int)
z_pool = z_sum / max(n_eff, 1)
print("  Pair enrichment (aggregated z; + co-localise, - exclude):")
for (A,B) in key_pairs:
    ia, ib = TYPES.index(A), TYPES.index(B)
    print(f"    {A:6s} -> {B:6s} : z={z_pool[ia,ib]:+.2f}   significant in {int(sig_cnt[ia,ib])}/{n_eff} sections")

# ---- (6) front vs core composition ----
print("\n=== (6) Invasive-front vs inner-site composition ===")
allprop = np.vstack([m["prop"] for m in meta])
for reg in ["invasive_front","inner_site"]:
    mm = reg_all == reg
    if mm.sum():
        print(f"  [{reg:14s}] n={int(mm.sum()):5d}  Tumor={allprop[mm,TYPES.index('Tumor')].mean():.3f}  "
              f"Mac={allprop[mm,TYPES.index('Mac')].mean():.3f}  NK={allprop[mm,TYPES.index('NK')].mean():.3f}  "
              f"Tcell={allprop[mm,TYPES.index('Tcell')].mean():.3f}")

# ---------- figures ----------
def plot_histo(si, field, fname, cmap, vmax=None):
    m = meta[si]; ad=m["ad"]; coords=m["coords"]; labels=m["labels"]; prop=m["prop"]; dv=m["dv"]
    fig, ax = plt.subplots(figsize=(5,5))
    sf = ad.uns.get("scalefactors", {})
    hires = os.path.join(RAW, m["name"], "spatial", "tissue_hires_image.png")
    try:
        if os.path.exists(hires) and sf:
            from matplotlib.image import imread
            img = imread(hires); sc_f = sf.get("tissue_hires_scalef", 1.0)
            c = coords * sc_f
            ax.imshow(img)
            if field == "label":
                cm = plt.get_cmap("tab10")
                ax.scatter(c[:,0], c[:,1], s=14, c=labels, cmap=cm, vmin=-0.5, vmax=9.5, edgecolor="none")
            elif field == "DEPTOR":
                ax.scatter(c[:,0], c[:,1], s=14, c=dv, cmap=cmap, vmin=0, vmax=vmax, edgecolor="none")
            else:
                v = prop[:, TYPES.index(field)]
                ax.scatter(c[:,0], c[:,1], s=14, c=v, cmap=cmap, vmin=0, vmax=vmax, edgecolor="none")
            ax.set_xticks([]); ax.set_yticks([])
        else:
            raise RuntimeError("no image")
    except Exception:
        v = labels if field=="label" else (dv if field=="DEPTOR" else prop[:, TYPES.index(field)])
        if v.ndim == 0: v = np.full(len(labels), v)
        ax.scatter(coords[:,0], coords[:,1], s=12, c=v, cmap=cmap, vmin=0, vmax=vmax)
    ax.set_title(f"{m['name']}  {m['region']}\n{field}", fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, fname), dpi=130); plt.close()

for si, m in enumerate(meta):
    n = m["name"]
    plot_histo(si, "label", f"spatial_{n}_label.png", "tab10")
    plot_histo(si, "Tumor", f"spatial_{n}_Tumor.png", "Greens")
    plot_histo(si, "NK",    f"spatial_{n}_NK.png", "viridis")
    plot_histo(si, "Mac",   f"spatial_{n}_Mac.png", "Reds")
    plot_histo(si, "DEPTOR",f"spatial_{n}_DEPTOR.png", "magma", vmax=1.0)

# enrichment heatmap
fig, ax = plt.subplots(figsize=(7,6))
im = ax.imshow(z_pool, cmap="RdBu_r", vmin=-5, vmax=5)
ax.set_xticks(range(K_T)); ax.set_yticks(range(K_T))
ax.set_xticklabels(TYPES, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(TYPES, fontsize=8)
for i in range(K_T):
    for j in range(K_T):
        if abs(z_pool[i,j]) >= 1.5:
            ax.text(j, i, f"{z_pool[i,j]:+.1f}", ha="center", va="center", fontsize=6)
ax.set_xlabel("neighbour cell type"); ax.set_ylabel("seed cell type")
ax.set_title("GSE181300 cell-type neighbourhood enrichment\n(aggregated z across sections; + co-localise, - exclude)")
fig.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout(); plt.savefig(os.path.join(OUT,"spatial_enrichment_heatmap.png"), dpi=140); plt.close()

# DEPTOR localisation bar
if n_pres:
    fig, ax = plt.subplots(figsize=(5.5,4))
    vals = [f_any, f_pres]
    ax.bar(["all spots","DEPTOR-present"], vals, color=["#999999","#8c564b"])
    for i,v in enumerate(vals): ax.text(i, v+0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylabel("fraction labelled Tumor (malignant epithelium)")
    ax.set_ylim(0, max(vals)*1.25 if max(vals)>0 else 1)
    ax.set_title(f"DEPTOR-present spots map onto malignant epithelium\npooled n(DEPTOR+)={n_pres}")
    plt.tight_layout(); plt.savefig(os.path.join(OUT,"spatial_DEPTOR_localisation.png"), dpi=140); plt.close()

# front vs core composition
fig, ax = plt.subplots(figsize=(7.5,4))
means = {}
for reg in ["invasive_front","inner_site"]:
    mm = reg_all == reg
    means[reg] = [allprop[mm, TYPES.index(t)].mean() if mm.sum() else 0 for t in TYPES]
x = np.arange(K_T); w = 0.35
ax.bar(x-w/2, means["invasive_front"], w, label="invasive_front", color="#d62728")
ax.bar(x+w/2, means["inner_site"], w, label="inner_site", color="#1f77b4")
ax.set_xticks(x); ax.set_xticklabels(TYPES, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("mean cell-type proportion")
ax.set_title("Front vs inner-site composition (GSE181300)")
ax.legend(fontsize=8); plt.tight_layout()
plt.savefig(os.path.join(OUT,"spatial_front_core_composition.png"), dpi=140); plt.close()

# DEPTOR-present neighbour immunity bar
if len(P_nk):
    fig, ax = plt.subplots(figsize=(6,4))
    xs = np.arange(2); w = 0.35
    ax.bar(xs-w/2, [P_nk.mean(), P_mac.mean()], w, label=f"DEPTOR-present tumor (n={len(P_nk)})", color="#d62728")
    ax.bar(xs+w/2, [A_nk.mean(), A_mac.mean()], w, label=f"DEPTOR-absent tumor (n={len(A_nk)})", color="#7f7f7f")
    ax.set_xticks(xs); ax.set_xticklabels(["neighbour NK","neighbour Mac"])
    ax.set_ylabel("mean neighbour proportion")
    ax.set_title("Neighbour immunity of tumor spots by DEPTOR presence")
    ax.legend(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(OUT,"spatial_DEPTOR_tumor_neighbour.png"), dpi=140); plt.close()

res.to_csv(os.path.join(OUT,"spatial_summary.csv"), index=False)
print("\nSaved to", OUT, "DONE.")
