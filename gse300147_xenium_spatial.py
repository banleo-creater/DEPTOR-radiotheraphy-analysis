#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GSE300147 — Xenium single-cell spatial transcriptomics of HNSCC
(10 patients, 17 sections, p16/HPV annotated).

VERIFIED LIMITATION: the 476-gene panel does NOT contain DEPTOR (nor DEPDC6,
EPYC, MTOR, RPTOR, RICTOR). DEPTOR itself therefore CANNOT be validated in this
dataset. What it CAN test — at true single-cell resolution — is the DOWNSTREAM
spatial claim of the DEPTOR hypothesis:

  * Do NK cells sit close to, or away from, myeloid (Mac / DC) cells?
  * Are tumour (EPCAM/EGFR-high) cells NK-depleted in their immediate
    neighbourhood (NK exclusion)?
  * Does any of this differ by HPV/p16 status?

Method
------
Per section, one cKDTree gives each cell's k nearest neighbours (k=60).
Because nearest-neighbour geometry depends only on coordinates (not labels),
the tree is built ONCE; label permutation then only re-indexes the precomputed
neighbour table. For each ordered pair (A->B):
    - mean distance from sampled A cells to their nearest B neighbour
    - fraction of B cells among the k nearest neighbours of A cells
compared with a null from label permutation (NPERM shuffles) -> z-scores.

Only the two small files per sample are used (cell_feature_matrix.h5 +
cells.parquet); transcripts / boundaries / morphology are skipped.
"""
import os, re, warnings
import numpy as np, pandas as pd
import h5py, scanpy as sc
from scipy.sparse import csc_matrix
from scipy.spatial import cKDTree
from scipy.stats import norm as normdist, mannwhitneyu, ttest_1samp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore"); sc.settings.verbosity = 0

BASE = "WorkBuddy/2026-10-17-11-15-46"
RAW  = os.path.join(BASE, "data", "gse300147", "raw")
OUT  = os.path.join(BASE, "data", "results_xenium"); os.makedirs(OUT, exist_ok=True)

KNN_K   = 60     # neighbours precomputed per cell
NPERM   = 199    # label permutations
SAMPLE_A = 300   # A-cells sampled per pair (keeps density intact)

# p16/HPV: p16 Positive (oropharynx) = HPV+; Negative or N/A (oral cavity /
# larynx / hypopharynx, where p16 is not routinely scored) treated as HPV-.
P16 = {
 "GSM9054471":"HPV+","GSM9054472":"HPV+","GSM9054473":"HPV+","GSM9054474":"HPV+",
 "GSM9054475":"HPV+","GSM9054476":"HPV+","GSM9054477":"HPV+",
 "GSM9054486":"HPV+","GSM9054487":"HPV+",
 "GSM9054478":"HPV-","GSM9054479":"HPV-","GSM9054480":"HPV-","GSM9054481":"HPV-",
 "GSM9054482":"HPV-","GSM9054483":"HPV-","GSM9054484":"HPV-","GSM9054485":"HPV-",
}
PATIENT = {
 "GSM9054471":"P1","GSM9054472":"P9","GSM9054473":"P9","GSM9054474":"P12","GSM9054475":"P12",
 "GSM9054476":"P13","GSM9054477":"P13","GSM9054486":"P28","GSM9054487":"P28",
 "GSM9054478":"P15","GSM9054479":"P17","GSM9054480":"P17","GSM9054481":"P19",
 "GSM9054482":"P19","GSM9054483":"P20","GSM9054484":"P20","GSM9054485":"P23",
}

# markers actually present in this panel (verified against the h5)
MARKERS = {
    "NK":     ["NKG7","GNLY","KLRD1","KLRC1","FCGR3A"],
    "Mac":    ["CD68","CD163","MRC1","VSIG4","CD14","AIF1"],
    "DC":     ["LAMP3","CD1C","FCER1A"],
    "Tcell":  ["CD3D","CD3E","CD8A","CD4","IL7R","CD2"],
    "Bcell":  ["MS4A1","CD79A"],
    "Tumor":  ["EPCAM","EGFR"],
    "Stroma": ["ACTA2","PDGFRA"],
    "Endo":   ["PECAM1","VWF"],
}
TYPES = list(MARKERS.keys())
iNK, iMac, iDC, iTum = (TYPES.index(t) for t in ["NK","Mac","DC","Tumor"])
KEY_PAIRS = [("NK","Mac"),("Mac","NK"),("NK","DC"),("DC","NK"),
             ("Tumor","NK"),("NK","Tumor"),("Tumor","Mac"),("Mac","Tumor"),
             ("Tumor","Tcell"),("Tcell","Tumor")]

def read_parquet_any(path):
    for engine in ("pyarrow", "fastparquet"):
        try:
            return pd.read_parquet(path, engine=engine)
        except Exception:
            continue
    try:
        import duckdb
        return duckdb.sql(f"SELECT * FROM read_parquet('{path}')").df()
    except Exception as e:
        raise RuntimeError(f"no parquet engine for {path}: {e}")

def load_sample(gsm):
    d = os.path.join(RAW, gsm)
    h5p, pq = os.path.join(d, "cell_feature_matrix.h5"), os.path.join(d, "cells.parquet")
    if not (os.path.exists(h5p) and os.path.exists(pq)):
        return None
    with h5py.File(h5p, "r") as f:
        names  = [n.decode() for n in f["matrix"]["features"]["name"][:]]
        ftype  = [n.decode() for n in f["matrix"]["features"]["feature_type"][:]]
        shape  = f["matrix"]["shape"][:]              # (genes, cells)
        barcodes = [b.decode() for b in f["matrix"]["barcodes"][:]]
        indptr, indices, data = (f["matrix"][k][:] for k in ("indptr","indices","data"))
    keep = [i for i, (n, t) in enumerate(zip(names, ftype))
            if t == "Gene Expression" and not re.search(r"_(TRA|TRB)$", n)]
    M = csc_matrix((data, indices, indptr), shape=tuple(shape)).tocsr().T.tocsr()
    M = M[:, keep].tocsr()
    ad = sc.AnnData(M.astype(np.float32))
    ad.obs_names = barcodes
    ad.var_names = pd.Index([names[i].upper() for i in keep]); ad.var_names_make_unique()
    cells = read_parquet_any(pq)
    ccol = "cell_id" if "cell_id" in cells.columns else cells.columns[0]
    cells = cells.set_index(ccol)
    common = [b for b in ad.obs_names if b in cells.index]
    ad = ad[common].copy()
    xcol = next((c for c in cells.columns if c.lower() in ("x_centroid", "x")), None)
    ycol = next((c for c in cells.columns if c.lower() in ("y_centroid", "y")), None)
    ad.obs["x"] = cells.loc[common, xcol].values.astype(float)
    ad.obs["y"] = cells.loc[common, ycol].values.astype(float)
    for cand in ("transcript_counts", "transcript_count", "total_counts"):
        if cand in cells.columns:
            ad.obs["n_transcripts"] = cells.loc[common, cand].values
            break
    else:
        ad.obs["n_transcripts"] = np.asarray(ad.X.sum(axis=1)).ravel()
    return ad

def score(ad, mk):
    vidx = {str(v): i for i, v in enumerate(ad.var_names)}
    idxs = [vidx[m.upper()] for m in mk if m.upper() in vidx]
    if not idxs:
        return np.zeros(ad.n_obs)
    s = ad.X[:, idxs].mean(axis=1)
    return np.asarray(s.todense()).ravel() if hasattr(s, "todense") else np.asarray(s).ravel()

# ---------------- spatial statistics (fast: tree built once) ----------------
def precompute(coords, k=KNN_K):
    tree = cKDTree(coords)
    k_eff = min(k, len(coords))
    d, idx = tree.query(coords, k=k_eff)
    return idx, d

def pair_stat(nn_idx, nn_dist, lab, A, B, rng):
    """mean distance A->nearest B, and fraction of B among k nearest neighbours."""
    ia_all = np.where(lab == A)[0]
    if ia_all.size == 0:
        return np.nan, np.nan
    ia = rng.choice(ia_all, SAMPLE_A, replace=False) if ia_all.size > SAMPLE_A else ia_all
    nb = lab[nn_idx[ia]]                       # (nA, k)
    isB = (nb == B)
    first = isB.argmax(axis=1)
    found = isB[np.arange(ia.size), first]
    dd = nn_dist[ia][np.arange(ia.size), first]
    nn = float(np.mean(dd[found])) if found.any() else np.nan
    return nn, float(isB.mean())

def perm_stats(nn_idx, nn_dist, labels, A, B, nperm=NPERM, seed=0):
    rng = np.random.default_rng(seed)
    obs_nn, obs_fr = pair_stat(nn_idx, nn_dist, labels, A, B, rng)
    if np.isnan(obs_nn):
        return obs_nn, np.nan, obs_fr, np.nan
    pnn, pfr = [], []
    lp = labels.copy()
    for _ in range(nperm):
        rng.shuffle(lp)
        a, b = pair_stat(nn_idx, nn_dist, lp, A, B, rng)
        pnn.append(a); pfr.append(b)
    pnn = np.array(pnn, dtype=float); pfr = np.array(pfr, dtype=float)
    zn = (obs_nn - np.nanmean(pnn)) / (np.nanstd(pnn) + 1e-12)
    zf = (obs_fr - np.nanmean(pfr)) / (np.nanstd(pfr) + 1e-12)
    return obs_nn, zn, obs_fr, zf

# ---------------- main ----------------
print("Loading Xenium sections ...")
records, meta = [], []
for gsm in sorted(P16):
    try:
        ad = load_sample(gsm)
    except Exception as e:
        print(f"  SKIP {gsm}: {e}"); continue
    if ad is None:
        print(f"  SKIP {gsm}: files missing"); continue
    ad = ad[ad.obs["n_transcripts"].values >= 5].copy()      # QC as in source paper
    sc.pp.normalize_total(ad, target_sum=1e4); sc.pp.log1p(ad)
    S = np.column_stack([score(ad, MARKERS[t]) for t in TYPES])
    labels = np.argmax(S, axis=1)
    coords = np.column_stack([ad.obs["x"].values, ad.obs["y"].values])
    nn_idx, nn_dist = precompute(coords)
    rec = dict(gsm=gsm, patient=PATIENT.get(gsm, "NA"), hpv=P16[gsm], n_cells=int(ad.n_obs),
               n_NK=int((labels == iNK).sum()), n_Mac=int((labels == iMac).sum()),
               n_DC=int((labels == iDC).sum()), n_Tumor=int((labels == iTum).sum()),
               frac_NK=float((labels == iNK).mean()), frac_Mac=float((labels == iMac).mean()),
               frac_Tumor=float((labels == iTum).mean()))
    for (a, b) in KEY_PAIRS:
        A, B = TYPES.index(a), TYPES.index(b)
        onn, zn, ofr, zf = perm_stats(nn_idx, nn_dist, labels, A, B, seed=hash(gsm) % 10000)
        rec[f"nn_{a}_{b}"] = onn; rec[f"z_{a}_{b}"] = zn
        rec[f"fr_{a}_{b}"] = ofr; rec[f"zf_{a}_{b}"] = zf
    records.append(rec)
    meta.append(dict(gsm=gsm, ad=ad, labels=labels, coords=coords, hpv=P16[gsm]))
    print(f"  {gsm} {PATIENT.get(gsm):4s} {P16[gsm]} cells={ad.n_obs:6d} "
          f"NK={rec['n_NK']:5d} Mac={rec['n_Mac']:5d} Tumor={rec['n_Tumor']:5d} | "
          f"NK->Mac {rec['nn_NK_Mac']:.1f}um (z={rec['z_NK_Mac']:+.2f}) | "
          f"Tumor->NK {rec['nn_Tumor_NK']:.1f}um (z={rec['z_Tumor_NK']:+.2f})")

res = pd.DataFrame(records)
res.to_csv(os.path.join(OUT, "xenium_summary.csv"), index=False)
print("\n=== Per-section summary ===")
cols = ["gsm","patient","hpv","n_cells","n_NK","n_Mac","n_Tumor","frac_NK","frac_Mac",
        "nn_NK_Mac","z_NK_Mac","nn_Tumor_NK","z_Tumor_NK","nn_Mac_NK","z_Mac_NK"]
print(res[cols].to_string(index=False))

print("\n=== Aggregated (z vs permutation; + farther/excluded, - closer/co-localised) ===")
for col, name in [("z_NK_Mac","NK -> Mac"),("z_Mac_NK","Mac -> NK"),
                  ("z_NK_DC","NK -> DC"),("z_DC_NK","DC -> NK"),
                  ("z_Tumor_NK","Tumor -> NK"),("z_NK_Tumor","NK -> Tumor"),
                  ("z_Tumor_Mac","Tumor -> Mac"),("z_Mac_Tumor","Mac -> Tumor"),
                  ("z_Tumor_Tcell","Tumor -> Tcell")]:
    v = res[col].dropna()
    if len(v) == 0: continue
    t, p = ttest_1samp(v, 0)
    print(f"  {name:16s} mean z={v.mean():+.2f}  n={len(v)} sections  p={p:.3g}  "
          f"(z<0 in {int((v<0).sum())}/{len(v)})")

# tumour NK exclusion: exact nearest-NK distance, tumour vs non-tumour cells
print("\n=== Tumour-cell NK exclusion (exact nearest-neighbour distance) ===")
d_t, d_n = [], []
for m in meta:
    labels, coords = m["labels"], m["coords"]
    ib = np.where(labels == iNK)[0]
    if ib.size == 0: continue
    tree = cKDTree(coords[ib])
    tum = np.where(labels == iTum)[0]
    non = np.where(labels != iTum)[0]
    if tum.size: d_t.append(tree.query(coords[tum], k=1)[0])
    if non.size: d_n.append(tree.query(coords[non], k=1)[0])
if d_t and d_n:
    dt, dn = np.concatenate(d_t), np.concatenate(d_n)
    print(f"  tumour cells     : median {np.median(dt):.1f} um (n={dt.size})")
    print(f"  non-tumour cells : median {np.median(dn):.1f} um (n={dn.size})")
    u, p = mannwhitneyu(dt, dn, alternative="greater")
    print(f"  -> tumour cells FARTHER from nearest NK: MWU p={p:.3g}")

print("\n=== HPV stratification ===")
for h in ["HPV+","HPV-"]:
    s = res[res.hpv == h]
    if not len(s): continue
    print(f"  [{h}] sections={len(s)} frac_NK={s.frac_NK.mean():.4f} frac_Mac={s.frac_Mac.mean():.4f} "
          f"nn(Tumor->NK)={s.nn_Tumor_NK.mean():.1f}um z={s.z_Tumor_NK.mean():+.2f} "
          f"z(NK->Mac)={s.z_NK_Mac.mean():+.2f}")
try:
    a = res.loc[res.hpv == "HPV+", "nn_Tumor_NK"].dropna()
    b = res.loc[res.hpv == "HPV-", "nn_Tumor_NK"].dropna()
    if len(a) and len(b):
        u, p = mannwhitneyu(a, b)
        print(f"  Tumor->NK distance HPV+ vs HPV-: MWU p={p:.3g}")
except Exception as e:
    print("  strat err", e)

# ---------------- figures ----------------
def plot_types(m, fname):
    labels, coords = m["labels"], m["coords"]
    fig, ax = plt.subplots(figsize=(6, 6))
    cm = plt.get_cmap("tab10")
    for t in range(len(TYPES)):
        s = labels == t
        if s.sum() == 0: continue
        ax.scatter(coords[s, 0], coords[s, 1], s=1.0, color=cm(t % 10), label=TYPES[t], alpha=0.6)
    ax.set_aspect("equal"); ax.set_xlabel("x (um)"); ax.set_ylabel("y (um)")
    ax.set_title(f"{m['gsm']} {PATIENT.get(m['gsm'],'')} {m['hpv']}  cells={m['ad'].n_obs}")
    ax.legend(markerscale=8, fontsize=7, loc="upper right")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, fname), dpi=130); plt.close()

for m in meta:
    plot_types(m, f"xenium_{m['gsm']}_types.png")

if len(res):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax_i, col in enumerate(["frac_NK", "nn_Tumor_NK", "z_NK_Mac"]):
        ax = axes[ax_i]
        vals = [res.loc[res.hpv == h, col].dropna().values for h in ["HPV+", "HPV-"]]
        ax.boxplot(vals, tick_labels=["HPV+", "HPV-"], showfliers=False)
        for i, v in enumerate(vals):
            ax.scatter(np.full(v.size, i + 1) + np.random.default_rng(0).normal(0, .04, v.size),
                       v, s=16, alpha=.8)
        ax.set_title(col); ax.axhline(0, color="k", lw=.6)
    plt.suptitle("GSE300147: HPV stratification of NK / myeloid spatial metrics")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "xenium_hpv_strat.png"), dpi=140); plt.close()

    zc = [c for c in res.columns if c.startswith("z_")]
    fig, ax = plt.subplots(figsize=(max(6, len(zc) * .55), 4))
    vals = [res[c].dropna().values for c in zc]
    ax.boxplot(vals, tick_labels=[c[2:] for c in zc], showfliers=False)
    ax.axhline(0, color="k", lw=.8)
    ax.set_ylabel("z vs permutation"); ax.set_xticklabels([c[2:] for c in zc], rotation=45, ha="right", fontsize=7)
    ax.set_title("GSE300147: nearest-neighbour z per cell-type pair\n(+ farther than random = spatial exclusion)")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "xenium_pair_z_box.png"), dpi=140); plt.close()

print("\nSaved to", OUT, "DONE.")
