#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GSE300147 (Xenium HNSCC, 17 sections) — MYELOID-SUBTYPE RESOLVED spatial
colocalisation of the myeloid–NK axis.

HARD CONSTRAINT (verified against the h5 feature table):
    DEPTOR is NOT in this 476-gene panel (nor DEPDC6 / MTOR / RPTOR / RICTOR,
    nor LYZ / CSF1R). A DEPTOR spatial test or an mTORC1-activity proxy is
    therefore IMPOSSIBLE here. What this script tests is the DOWNSTREAM
    ecological claim of the DEPTOR hypothesis:

        "a macrophage-dominated myeloid niche spatially excludes NK cells"

    and, critically, WHICH myeloid subset mediates it:
        M2-like  : CD163, MRC1, VSIG4
        M1-like  : CD68, CD14, AIF1
        cDC      : LAMP3, CD1C, FCER1A

Metrics (per ordered pair A->B, per section)
-------------------------------------------
1. RADIAL ENRICHMENT CURVE — mean number of B cells within radius r of an A
   cell, r = 25..250 um, compared with a label-permutation null:
       enrichment(r) = observed / expected      (>1 = colocalised, <1 = excluded)
       z(r)          = (observed - mean_null) / sd_null
   Because nearest-neighbour geometry depends only on coordinates, the tree is
   queried ONCE per (section, A-set); permutations only relabel.
2. NEAREST-NEIGHBOUR z (consistent with the previous 8-class run) — mean
   distance from A to its nearest B cell vs permutation null.
"""
import os, re, warnings, json
import numpy as np, pandas as pd
import h5py
from scipy.sparse import csc_matrix
from scipy.spatial import cKDTree
from scipy.stats import ttest_1samp, ttest_rel, mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

BASE = "WorkBuddy/2026-10-17-11-15-46"
RAW  = os.path.join(BASE, "data", "gse300147", "raw")
OUT  = os.path.join(BASE, "data", "results_xenium_mnk")
os.makedirs(OUT, exist_ok=True)

# ---- parameters ----
K_NEIGH   = 2000    # neighbours queried per sampled A cell (covers >=250 um)
SAMPLE_A  = 600     # A cells sampled per pair
NPERM     = 199
RADII     = np.array([25, 50, 75, 100, 150, 200, 250], dtype=float)
MIN_CELLS = 50      # a type needs >= this many cells in a section to be tested
SEED      = 7

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

# ---- STAGE 1: lineage typing with the PREVIOUSLY VALIDATED 8-class markers ----
# (unchanged from gse300147_xenium_spatial.py so NK counts stay comparable)
LIN = {
    "NK":     ["NKG7", "GNLY", "KLRD1", "KLRC1", "FCGR3A"],
    "Mac":    ["CD68", "CD163", "MRC1", "VSIG4", "CD14", "AIF1"],
    "DC":     ["LAMP3", "CD1C", "FCER1A"],
    "Tcell":  ["CD3D", "CD3E", "CD8A", "CD4", "IL7R", "CD2"],
    "Bcell":  ["MS4A1", "CD79A"],
    "Tumor":  ["EPCAM", "EGFR"],
    "Stroma": ["ACTA2", "PDGFRA"],
    "Endo":   ["PECAM1", "VWF"],
}
LIN_TYPES = list(LIN.keys())

# ---- STAGE 2: split the Mac lineage into M2-like / M1-like WITHIN macrophages ----
# Comparing the two sub-scores inside already-macrophage cells avoids the
# argmax dilution that happens if M1/M2 compete as separate lineage classes.
M2_MK, M1_MK = ["CD163", "MRC1", "VSIG4"], ["CD68", "CD14", "AIF1"]

# final classes actually used in the spatial statistics
TYPES = ["NK", "M2", "M1", "DC", "Tcell", "Bcell", "Tumor", "Stroma", "Endo"]
TIDX = {t: i for i, t in enumerate(TYPES)}

# ordered pairs of interest: (A, B, colour-group)
PAIRS = [
    ("NK", "M2"), ("NK", "M1"), ("NK", "DC"),
    ("M2", "NK"), ("M1", "NK"), ("DC", "NK"),
    ("NK", "Tumor"), ("Tumor", "NK"), ("Tumor", "Tcell"),
]
# pairs re-run INSIDE the non-tumour (stroma/immune) compartment only, so that
# the tumour-vs-stroma compartment split cannot drive myeloid-NK colocalisation.
NT_PAIRS = [("NK", "M2"), ("NK", "M1"), ("NK", "DC"),
            ("M2", "NK"), ("M1", "NK"), ("DC", "NK")]


def read_parquet_any(path):
    for engine in ("pyarrow", "fastparquet"):
        try:
            return pd.read_parquet(path, engine=engine)
        except Exception:
            continue
    raise RuntimeError(f"no parquet engine for {path}")


def load_sample(gsm):
    d = os.path.join(RAW, gsm)
    h5p, pq = os.path.join(d, "cell_feature_matrix.h5"), os.path.join(d, "cells.parquet")
    if not (os.path.exists(h5p) and os.path.exists(pq)):
        return None
    with h5py.File(h5p, "r") as f:
        names = [n.decode() for n in f["matrix"]["features"]["name"][:]]
        ftype = [n.decode() for n in f["matrix"]["features"]["feature_type"][:]]
        shape = f["matrix"]["shape"][:]
        barcodes = [b.decode() for b in f["matrix"]["barcodes"][:]]
        indptr, indices, data = (f["matrix"][k][:] for k in ("indptr", "indices", "data"))
    keep = [i for i, (n, t) in enumerate(zip(names, ftype))
            if t == "Gene Expression" and not re.search(r"_(TRA|TRB)$", n)]
    M = csc_matrix((data, indices, indptr), shape=tuple(shape)).tocsr().T.tocsr()[:, keep].tocsr()
    X = M.astype(np.float32)
    # library-size normalise + log1p (avoids a scanpy dependency here)
    tot = np.asarray(X.sum(axis=1)).ravel()
    tot[tot == 0] = 1.0
    X = X.multiply(1e4 / tot[:, None]).tocsr()
    X.data = np.log1p(X.data)
    vnames = pd.Index([names[i].upper() for i in keep])
    cells = read_parquet_any(pq)
    ccol = "cell_id" if "cell_id" in cells.columns else cells.columns[0]
    cells = cells.set_index(ccol)
    common = [b for b in barcodes if b in cells.index]
    cidx = {b: i for i, b in enumerate(barcodes)}
    rows = np.array([cidx[b] for b in common])
    X = X[rows]
    xcol = next(c for c in cells.columns if c.lower() in ("x_centroid", "x"))
    ycol = next(c for c in cells.columns if c.lower() in ("y_centroid", "y"))
    xy = cells.loc[common, [xcol, ycol]].values.astype(float)
    nt = None
    for cand in ("transcript_counts", "transcript_count", "total_counts"):
        if cand in cells.columns:
            nt = cells.loc[common, cand].values.astype(float)
            break
    if nt is None:
        nt = np.asarray(X.sum(axis=1)).ravel()
    return dict(X=X, vnames=vnames, xy=xy, nt=nt, gsm=gsm)


def type_labels(s):
    """Two-stage typing: validated 8-class lineage, then Mac -> M2/M1 split."""
    vidx = {str(v): i for i, v in enumerate(s["vnames"])}
    X = s["X"]

    def sc(mk):
        idx = [vidx[m] for m in mk if m in vidx]
        return np.asarray(X[:, idx].mean(axis=1)).ravel() if idx else np.zeros(X.shape[0])

    S = np.column_stack([sc(LIN[t]) for t in LIN_TYPES])
    lin = np.argmax(S, axis=1)
    labels = np.empty(X.shape[0], dtype=np.int8)
    lmap = {"NK": "NK", "DC": "DC", "Tcell": "Tcell", "Bcell": "Bcell",
            "Tumor": "Tumor", "Stroma": "Stroma", "Endo": "Endo"}
    for li, lt in enumerate(LIN_TYPES):
        if lt in lmap:
            labels[lin == li] = TIDX[lmap[lt]]
    mac = lin == LIN_TYPES.index("Mac")
    m2, m1 = sc(M2_MK), sc(M1_MK)
    labels[mac] = np.where(m2[mac] > m1[mac], TIDX["M2"], TIDX["M1"])
    return labels, mac


def query_A(tree, coords, ia, k):
    k_eff = int(min(k, coords.shape[0]))
    d, idx = tree.query(coords[ia], k=k_eff, workers=-1)
    return np.atleast_2d(idx), np.atleast_2d(d)


def curve_stats(nn_idx, nn_dist, labels, B, nperm, rng):
    """Return observed counts, expected counts, z and enrichment per radius."""
    nb = labels[nn_idx]                       # (nA, K)
    isB = (nb == B)
    obs = np.array([((nn_dist <= r) & isB).sum(axis=1).mean() for r in RADII])
    # saturation guard: if the K-th neighbour is nearer than r, counts are capped
    sat = np.array([(nn_dist[:, -1] < r).mean() for r in RADII])
    perm = np.empty((nperm, RADII.size))
    lp = labels.copy()
    for p in range(nperm):
        rng.shuffle(lp)
        pb = (lp[nn_idx] == B)
        perm[p] = [((nn_dist <= r) & pb).sum(axis=1).mean() for r in RADII]
    exp = perm.mean(axis=0)
    sd = perm.std(axis=0, ddof=1)
    z = (obs - exp) / (sd + 1e-12)
    enr = obs / (exp + 1e-12)
    return obs, exp, z, enr, sat


def nn_stats(nn_idx, nn_dist, labels, B, nperm, rng):
    """Nearest-B distance z (positive = farther than random = excluded)."""
    isB = (labels[nn_idx] == B)
    found = isB.any(axis=1)
    if not found.any():
        return np.nan, np.nan
    first = isB.argmax(axis=1)
    dd = nn_dist[np.arange(isB.shape[0]), first]
    obs = float(dd[found].mean())
    lp = labels.copy()
    perm = []
    for _ in range(nperm):
        rng.shuffle(lp)
        pb = (lp[nn_idx] == B)
        f2 = pb.any(axis=1)
        if not f2.any():
            continue
        perm.append(float(nn_dist[np.arange(pb.shape[0]), pb.argmax(axis=1)][f2].mean()))
    perm = np.array(perm)
    return obs, float((obs - perm.mean()) / (perm.std(ddof=1) + 1e-12))


# ------------------------------- main -------------------------------
print("=" * 78)
print("Xenium myeloid-subtype resolved NK colocalisation — GSE300147")
print("NOTE: DEPTOR is NOT in this panel; this is a DOWNSTREAM niche test.")
print("=" * 78)

rows, curves, meta = [], [], []
for gsm in sorted(P16):
    s = load_sample(gsm)
    if s is None:
        print(f"  SKIP {gsm}: files missing")
        continue
    keep = s["nt"] >= 5
    s["X"] = s["X"][keep]
    s["xy"] = s["xy"][keep]
    labels, _ = type_labels(s)
    coords = s["xy"]
    n = coords.shape[0]
    tree = cKDTree(coords)
    rng = np.random.default_rng(SEED + hash(gsm) % 1000)

    comp = {t: int((labels == TIDX[t]).sum()) for t in TYPES}
    rec = dict(gsm=gsm, patient=PATIENT.get(gsm, "NA"), hpv=P16[gsm], n_cells=n,
               **{f"n_{t}": comp[t] for t in TYPES},
               **{f"frac_{t}": comp[t] / n for t in TYPES})

    for (a, b) in PAIRS:
        A, B = TIDX[a], TIDX[b]
        ia_all = np.where(labels == A)[0]
        if ia_all.size < MIN_CELLS or comp[b] < MIN_CELLS:
            continue
        ia = rng.choice(ia_all, min(SAMPLE_A, ia_all.size), replace=False)
        nn_idx, nn_dist = query_A(tree, coords, ia, K_NEIGH)
        obs, exp, z, enr, sat = curve_stats(nn_idx, nn_dist, labels, B, NPERM, rng)
        nnd, nnz = nn_stats(nn_idx, nn_dist, labels, B, NPERM, rng)
        rec[f"nnd_{a}_{b}"] = nnd
        rec[f"nnz_{a}_{b}"] = nnz
        for j, r in enumerate(RADII):
            rec[f"enr_{a}_{b}_{int(r)}"] = enr[j] if sat[j] < 0.05 else np.nan
            rec[f"z_{a}_{b}_{int(r)}"] = z[j] if sat[j] < 0.05 else np.nan
        curves.append(dict(gsm=gsm, patient=PATIENT[gsm], hpv=P16[gsm], A=a, B=b,
                           nA=int(ia_all.size), nB=int(comp[b]),
                           **{f"r{int(r)}": enr[j] for j, r in enumerate(RADII)},
                           **{f"z{int(r)}": z[j] for j, r in enumerate(RADII)}))
    # ---- compartment-conditioned: re-run inside the NON-TUMOUR compartment ----
    ntmask = labels != TIDX["Tumor"]
    if ntmask.sum() > 500:
        nc = coords[ntmask]
        nlab = labels[ntmask]
        ntree = cKDTree(nc)
        for (a, b) in NT_PAIRS:
            A, B = TIDX[a], TIDX[b]
            ia_all = np.where(nlab == A)[0]
            if ia_all.size < MIN_CELLS or int((nlab == B).sum()) < MIN_CELLS:
                continue
            ia = rng.choice(ia_all, min(SAMPLE_A, ia_all.size), replace=False)
            nn_idx, nn_dist = query_A(ntree, nc, ia, K_NEIGH)
            obs, exp, z, enr, sat = curve_stats(nn_idx, nn_dist, nlab, B, NPERM, rng)
            nnd, nnz = nn_stats(nn_idx, nn_dist, nlab, B, NPERM, rng)
            rec[f"nt_nnd_{a}_{b}"] = nnd
            rec[f"nt_nnz_{a}_{b}"] = nnz
            for j, r in enumerate(RADII):
                rec[f"nt_enr_{a}_{b}_{int(r)}"] = enr[j] if sat[j] < 0.05 else np.nan
                rec[f"nt_z_{a}_{b}_{int(r)}"] = z[j] if sat[j] < 0.05 else np.nan

    rows.append(rec)
    meta.append(dict(gsm=gsm, labels=labels, coords=coords, hpv=P16[gsm],
                     patient=PATIENT[gsm]))
    print(f"  {gsm} {PATIENT[gsm]:4s} {P16[gsm]} n={n:6d} "
          f"NK={comp['NK']:5d} M2={comp['M2']:5d} M1={comp['M1']:5d} DC={comp['DC']:4d} "
          f"Tumor={comp['Tumor']:6d} | "
          f"NK->M2 enr@100={rec.get('enr_NK_M2_100', float('nan')):.2f} "
          f"(non-tum {rec.get('nt_enr_NK_M2_100', float('nan')):.2f})  "
          f"NK->M1 {rec.get('enr_NK_M1_100', float('nan')):.2f} "
          f"(non-tum {rec.get('nt_enr_NK_M1_100', float('nan')):.2f})")

res = pd.DataFrame(rows)
cur = pd.DataFrame(curves)
res.to_csv(os.path.join(OUT, "xenium_mnk_persection.csv"), index=False)
cur.to_csv(os.path.join(OUT, "xenium_mnk_curves.csv"), index=False)
print(f"\nsaved per-section table ({len(res)} sections) and {len(cur)} curves")

# --------------------------- aggregate inference ---------------------------
print("\n" + "=" * 78)
print("AGGREGATE — nearest-neighbour z (positive = farther than random = EXCLUDED)")
print("=" * 78)
agg = []
for (a, b) in PAIRS:
    c = f"nnz_{a}_{b}"
    if c not in res.columns:
        continue
    v = res[c].dropna()
    if len(v) < 3:
        continue
    t, p = ttest_1samp(v, 0)
    agg.append(dict(pair=f"{a}->{b}", n_sections=len(v), mean_z=v.mean(), sd=v.std(ddof=1),
                    t=t, p=p, n_excluded=int((v > 0).sum()),
                    n_colocalised=int((v < 0).sum())))
    print(f"  {a:6s} -> {b:6s}  mean z = {v.mean():+6.2f}  n={len(v):2d}  "
          f"p={p:9.3g}   (z>0 in {int((v>0).sum())}/{len(v)})")
aggdf = pd.DataFrame(agg)
aggdf.to_csv(os.path.join(OUT, "xenium_mnk_nnz_aggregate.csv"), index=False)

# paired contrasts: does NK avoid M2 more than M1? and DC?
print("\n" + "=" * 78)
print("PAIRED CONTRASTS within the same section (NK -> myeloid subset)")
print("=" * 78)
contr = []
for b1, b2 in [("M2", "M1"), ("M2", "DC"), ("M1", "DC")]:
    c1, c2 = f"nnz_NK_{b1}", f"nnz_NK_{b2}"
    if c1 not in res.columns or c2 not in res.columns:
        continue
    m = res[[c1, c2]].dropna()
    if len(m) < 3:
        continue
    t, p = ttest_rel(m[c1], m[c2])
    contr.append(dict(contrast=f"NK->{b1} minus NK->{b2}", n=len(m),
                      mean_diff=float((m[c1] - m[c2]).mean()), t=float(t), p=float(p),
                      n_b1_more_excluded=int(((m[c1] - m[c2]) > 0).sum())))
    print(f"  NK->{b1:3s} vs NK->{b2:3s}: mean diff = {(m[c1]-m[c2]).mean():+.2f}  "
          f"n={len(m)}  paired p={p:.4g}  "
          f"(NK further from {b1} in {int(((m[c1]-m[c2])>0).sum())}/{len(m)})")
pd.DataFrame(contr).to_csv(os.path.join(OUT, "xenium_mnk_paired_contrasts.csv"), index=False)

# enrichment at 100 um, aggregated
print("\n" + "=" * 78)
print("RADIAL ENRICHMENT (observed / permuted) at r = 100 um")
print("=" * 78)
enr_rows = []
for (a, b) in PAIRS:
    c = f"enr_{a}_{b}_100"
    if c not in res.columns:
        continue
    v = res[c].dropna()
    if len(v) < 3:
        continue
    t, p = ttest_1samp(v, 1.0)
    enr_rows.append(dict(pair=f"{a}->{b}", n=len(v), mean_enr=v.mean(),
                         ci_lo=v.mean() - 1.96 * v.std(ddof=1) / np.sqrt(len(v)),
                         ci_hi=v.mean() + 1.96 * v.std(ddof=1) / np.sqrt(len(v)),
                         p_vs_1=p))
    print(f"  {a:6s} -> {b:6s}  enrichment@100um = {v.mean():.3f} "
          f"(95%CI {v.mean()-1.96*v.std(ddof=1)/np.sqrt(len(v)):.3f}-"
          f"{v.mean()+1.96*v.std(ddof=1)/np.sqrt(len(v)):.3f})  n={len(v)}  p(vs 1)={p:.3g}")
pd.DataFrame(enr_rows).to_csv(os.path.join(OUT, "xenium_mnk_enrich100.csv"), index=False)

# HPV stratification
print("\n" + "=" * 78)
print("HPV STRATIFICATION")
print("=" * 78)
hpv_rows = []
for c, lab in [("nnz_NK_M2", "NK->M2 z"), ("nnz_NK_M1", "NK->M1 z"),
               ("nnz_NK_DC", "NK->DC z"), ("nnz_Tumor_NK", "Tumor->NK z"),
               ("frac_NK", "NK fraction"), ("frac_M2", "M2 fraction")]:
    if c not in res.columns:
        continue
    a = res.loc[res.hpv == "HPV+", c].dropna()
    b = res.loc[res.hpv == "HPV-", c].dropna()
    if len(a) < 3 or len(b) < 3:
        continue
    u, p = mannwhitneyu(a, b)
    hpv_rows.append(dict(metric=lab, hpv_pos=a.mean(), hpv_neg=b.mean(),
                         n_pos=len(a), n_neg=len(b), MWU_p=p))
    print(f"  {lab:14s} HPV+ = {a.mean():+.3f}  HPV- = {b.mean():+.3f}  "
          f"MWU p = {p:.4g}")
pd.DataFrame(hpv_rows).to_csv(os.path.join(OUT, "xenium_mnk_hpv.csv"), index=False)

# -------- compartment-conditioned (non-tumour) aggregation --------
print("\n" + "=" * 78)
print("COMPARTMENT-CONDITIONED: same statistics inside the NON-TUMOUR compartment")
print("(removes the tumour-nest vs stroma confound)")
print("=" * 78)
nt_rows = []
for (a, b) in NT_PAIRS:
    cz, ce = f"nt_nnz_{a}_{b}", f"nt_enr_{a}_{b}_100"
    if cz not in res.columns:
        continue
    v = res[cz].dropna()
    if len(v) >= 3:
        t, p = ttest_1samp(v, 0)
        nt_rows.append(dict(pair=f"{a}->{b}", kind="NN_z", n=len(v), est=v.mean(), p=p))
        print(f"  NN z   {a:6s} -> {b:6s}  mean z = {v.mean():+6.2f}  n={len(v):2d}  p={p:.3g}")
    v2 = res[ce].dropna() if ce in res.columns else pd.Series(dtype=float)
    if len(v2) >= 3:
        t, p = ttest_1samp(v2, 1.0)
        nt_rows.append(dict(pair=f"{a}->{b}", kind="enrich@100um", n=len(v2), est=v2.mean(), p=p))
        print(f"  enr100 {a:6s} -> {b:6s}  = {v2.mean():.3f}  n={len(v2):2d}  p(vs 1)={p:.3g}")
pd.DataFrame(nt_rows).to_csv(os.path.join(OUT, "xenium_mnk_nontumor.csv"), index=False)

print("\n  paired contrasts (non-tumour compartment):")
ntc = []
for b1, b2 in [("M2", "M1"), ("M2", "DC"), ("M1", "DC")]:
    c1, c2 = f"nt_nnz_NK_{b1}", f"nt_nnz_NK_{b2}"
    if c1 not in res.columns or c2 not in res.columns:
        continue
    m = res[[c1, c2]].dropna()
    if len(m) < 3:
        continue
    t, p = ttest_rel(m[c1], m[c2])
    ntc.append(dict(contrast=f"NK->{b1} minus NK->{b2} (non-tumour)", n=len(m),
                    mean_diff=float((m[c1] - m[c2]).mean()), p=float(p)))
    print(f"    NK->{b1:3s} vs NK->{b2:3s}: diff={(m[c1]-m[c2]).mean():+.2f}  "
          f"n={len(m)}  paired p={p:.4g}")
pd.DataFrame(ntc).to_csv(os.path.join(OUT, "xenium_mnk_nontumor_contrasts.csv"), index=False)

json.dump(dict(n_sections=int(len(res)), K_NEIGH=K_NEIGH, SAMPLE_A=SAMPLE_A,
               NPERM=NPERM, radii=RADII.tolist(), deptor_in_panel=False),
          open(os.path.join(OUT, "params.json"), "w"), indent=1)
print("\nDONE ->", OUT)
