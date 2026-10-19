#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cache the labelled cell coordinates of one representative Xenium section so the
figure script can draw panel F at print resolution without re-reading the .h5.

Typing is IDENTICAL to xenium_myeloid_nk_colocalization.py (two-stage: validated
8-class lineage argmax, then Mac split into M2/M1 by within-macrophage score),
copied here so that importing that module cannot re-run the whole analysis.

Output: data/results_xenium_mnk/xenium_GSM9054476_cells.csv
        (x_um, y_um, type)
"""
import os
import re
import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix

BASE = "WorkBuddy/2026-10-17-11-15-46"
RAW = os.path.join(BASE, "data", "gse300147", "raw")
OUT = os.path.join(BASE, "data", "results_xenium_mnk")
GSM = "GSM9054476"                      # P13, HPV+, the field used in the old panel F

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
M2_MK, M1_MK = ["CD163", "MRC1", "VSIG4"], ["CD68", "CD14", "AIF1"]
TYPES = ["NK", "M2", "M1", "DC", "Tcell", "Bcell", "Tumor", "Stroma", "Endo"]
TIDX = {t: i for i, t in enumerate(TYPES)}


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
    return dict(X=X, vnames=vnames, xy=xy, gsm=gsm)


def type_labels(s):
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
    return labels


s = load_sample(GSM)
lab = type_labels(s)
df = pd.DataFrame({"x_um": s["xy"][:, 0], "y_um": s["xy"][:, 1],
                   "type": [TYPES[i] for i in lab]})
p = os.path.join(OUT, f"xenium_{GSM}_cells.csv")
df.to_csv(p, index=False)
print("saved", p, df.shape)
print(df.type.value_counts().to_string())
print("x range %.0f-%.0f  y range %.0f-%.0f" %
      (df.x_um.min(), df.x_um.max(), df.y_um.min(), df.y_um.max()))
