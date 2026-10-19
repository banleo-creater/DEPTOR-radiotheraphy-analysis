#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EPIC and quanTIseq deconvolution of TCGA-HNSC, GSE65858 and GSE41613, and a
calibrated tumour-vs-immune split of bulk DEPTOR.

WHY
  The previous split used an in-house NNLS solver with a single-cell centroid
  reference.  EPIC (Racle 2017) and quanTIseq (Finotello 2019) are the two
  field-standard, calibration-aware methods, and both ship published signature
  matrices.  Here we re-implement their published algorithms in Python:

  EPIC
    * reference  : TRef (7 cell types: Bcells, CAFs, CD4_Tcells, CD8_Tcells,
                   Endothelial, Macrophages, NKcells) -- note there is NO
                   malignant column, so "otherCells" = cancer + uncharacterised.
    * scaling    : scaleCounts -> normalise bulk and reference to 1e6 over the
                   common genes, then subset to the 98 sigGenes.
    * objective  : weighted SSE, w_g = sum_c TRef_gc/(var_gc+1e-12), capped at
                   100 x median, subject to x >= 0 and sum(x) <= 1.
    * scaling    : mRNA proportions -> cell fractions via mRNA_cell_default.

  quanTIseq
    * reference  : TIL10 (10 immune types), mRNA scaling factors, noisy-gene and
                   TCGA-aberrant-gene blacklists.
    * preprocessing: un-log, quantile-normalise (arrays), TPM.
    * objective  : lsei (x >= 0, sum(x) <= 1) on the 2-norm-scaled system, then
                   divide by the mRNA scaling, renormalise, and append "Other".

  DEPTOR SPLIT (identical reference for both methods, so the methods differ only
  in the fractions):
    contribution_c = fraction_c * expm1(DEPTOR_c)   [DEPTOR_c from the GSE164690
                                                     single-cell centroids]
    DEPTOR_tumor_frac   = contribution_malignant / sum(contribution)
    DEPTOR_immune_frac  = sum(contribution_immune) / sum(contribution)

Outputs -> data/results_epic_quantiseq/
  <cohort>_EPIC_fractions.csv
  <cohort>_quantiseq_fractions.csv
  <cohort>_DEPTOR_split.csv
  validation_concordance.csv
"""
import os, sys, gzip, json
import numpy as np, pandas as pd
import rdata
from scipy.optimize import minimize
from scipy.stats import spearmanr, pearsonr

BASE = r"WorkBuddy/2026-10-17-11-15-46"
DATA = os.path.join(BASE, "data")
SIG = os.path.join(DATA, "signatures")
GEO = r"<PATH>"
OUT = os.path.join(DATA, "results_epic_quantiseq")
os.makedirs(OUT, exist_ok=True)

# ================================================================ signatures
print("loading signatures ...")
_t = rdata.read_rda(os.path.join(SIG, "TRef.rda"))["TRef"]
TREF = pd.DataFrame(_t["refProfiles"].values,
                    index=[str(g) for g in _t["refProfiles"].coords[_t["refProfiles"].dims[0]].values],
                    columns=[str(c) for c in _t["refProfiles"].coords[_t["refProfiles"].dims[1]].values])
TREF_VAR = pd.DataFrame(_t["refProfiles.var"].values, index=TREF.index, columns=TREF.columns)
EPIC_SIG = [str(g) for g in _t["sigGenes"]]
_m = rdata.read_rda(os.path.join(SIG, "mRNA_cell_default.rda"))["mRNA_cell_default"]
MRNA_CELL = pd.Series(_m.values, index=[str(x) for x in _m.coords[_m.dims[0]].values])
print(f"  TRef {TREF.shape}  ({', '.join(TREF.columns)})")
print(f"  EPIC sigGenes {len(EPIC_SIG)}; mRNA_cell {dict(MRNA_CELL.round(3))}")

TIL10 = pd.read_csv(os.path.join(SIG, "TIL10_signature.txt"), sep="\t", index_col=0)
TIL10.index = TIL10.index.astype(str)
_mrna = pd.read_csv(os.path.join(SIG, "TIL10_mRNA_scaling.txt"), sep="\t", header=None, names=["celltype", "scaling"])
TIL_SCALING = _mrna.set_index("celltype")["scaling"].reindex(TIL10.columns).values
RMGENES = set(pd.read_csv(os.path.join(SIG, "TIL10_rmgenes.txt"), sep="\t", header=None)[0].astype(str))
ABGENES = set(pd.read_csv(os.path.join(SIG, "TIL10_TCGA_aberrant_immune_genes.txt"), sep="\t", header=None)[0].astype(str))
print(f"  TIL10 {TIL10.shape}; rmgenes {len(RMGENES)}; aberrant {len(ABGENES)}")

# single-cell reference for DEPTOR per compartment
CENT = pd.read_csv(os.path.join(DATA, "gse164690_centroid_full.csv"), index_col=0)
DEPTOR_C = np.expm1(CENT["DEPTOR"].astype(float))          # CP10K scale
print(f"  centroid {CENT.shape}; DEPTOR per compartment: {DEPTOR_C.round(4).to_dict()}")

# ================================================================ cohort loaders
def _gpl_symbols(path, id_is_probe=True):
    rows, sym_col, intab = {}, None, False
    with gzip.open(path, "rt", errors="ignore") as f:
        for line in f:
            if line.startswith("!platform_table_begin"):
                intab = True; continue
            if line.startswith("!platform_table_end"):
                break
            if not intab:
                continue
            p = line.rstrip("\n").split("\t")
            if sym_col is None:
                cols = [c.strip().lower() for c in p]
                for cand in ("gene symbol", "symbol", "ilmn_gene", "gene_symbol"):
                    if cand in cols:
                        sym_col = cols.index(cand); break
                if sym_col is None:
                    sym_col = 2
                continue
            if len(p) <= sym_col:
                continue
            pid = p[0].strip().strip('"'); sym = p[sym_col].strip().strip('"')
            if pid and sym and sym not in ("", "---"):
                rows[pid] = sym.split("///")[0].strip()
    return rows


def _matrix(path):
    with gzip.open(path, "rt", errors="ignore") as f:
        lines = f.readlines()
    s = next(i for i, l in enumerate(lines) if l.startswith("!series_matrix_table_begin"))
    e = next(i for i, l in enumerate(lines) if l.startswith("!series_matrix_table_end"))
    hdr = [h.strip().strip('"') for h in lines[s + 1].rstrip("\n").split("\t")]
    ids, data = [], []
    for l in lines[s + 2:e]:
        p = l.rstrip("\n").split("\t")
        ids.append(p[0].strip().strip('"'))
        data.append([float(x) if x.strip() not in ("", "NA", "null", "NaN") else np.nan for x in p[1:]])
    return pd.DataFrame(data, index=ids, columns=hdr[1:])


def _collapse(raw, p2s):
    raw = raw.copy()
    raw["symbol"] = [p2s.get(i) for i in raw.index]
    raw = raw.dropna(subset=["symbol"])
    raw["symbol"] = raw["symbol"].replace({"DEPDC6": "DEPTOR"})
    return raw.groupby("symbol").max().T          # samples x genes


def load_cohort(name):
    if name == "TCGA":
        d = pd.read_csv(os.path.join(DATA, "tcga_hnsc_sig_genes.csv"), index_col=0)
        return d.apply(pd.to_numeric, errors="coerce")
    if name == "GSE65858":
        p2s = _gpl_symbols(os.path.join(DATA, "plat", "GPL10558.soft.gz"))
        raw = _matrix(os.path.join(GEO, "GSE65858", "GSE65858_series_matrix.txt.gz"))
        return _collapse(raw, p2s)
    if name == "GSE41613":
        p2s = _gpl_symbols(os.path.join(DATA, "plat", "GPL570.annot.gz"))
        raw = _matrix(os.path.join(GEO, "GSE41613", "GSE41613_series_matrix.txt.gz"))
        return _collapse(raw, p2s)
    raise ValueError(name)


# ================================================================ preprocessing
def to_linear_tpm(df, is_array):
    """un-log if needed, quantile-normalise arrays, then TPM (col sum 1e6)."""
    X = df.copy()
    if np.nanmax(X.values) < 50:
        X = 2.0 ** X
    X = X.clip(lower=0)
    if is_array:
        # quantile normalise across samples (columns) using the average sorted rank
        arr = X.values.astype(float)
        ranks = np.argsort(np.argsort(arr, axis=0), axis=0)
        # average value per rank across all samples
        sorted_arr = np.sort(arr, axis=0)
        avg = sorted_arr.mean(axis=1)
        out = np.empty_like(arr)
        for j in range(arr.shape[1]):
            out[:, j] = avg[ranks[:, j]]
        X = pd.DataFrame(out, index=X.index, columns=X.columns)
    X = X.div(X.sum(axis=1), axis=0) * 1e6
    return X.fillna(0)


# ================================================================ EPIC
def run_epic(bulk_tpm):
    """
    bulk_tpm : samples x genes, linear expression

    SCALE NOTE
      EPIC's scaleCounts normalises BOTH bulk and reference to 1e6 over the
      same common genes, so they are directly comparable even for the 257-gene
      TCGA subset (the absolute values differ from whole-transcriptome TPM, but
      the bulk/reference ratio is preserved, which is all the fit needs).
    """
    common = [g for g in TREF.index if g in bulk_tpm.columns]
    sig = [g for g in EPIC_SIG if g in bulk_tpm.columns and g in TREF.index]
    print(f"    EPIC: common genes {len(common)}, sigGenes present {len(sig)}/{len(EPIC_SIG)}")

    normB = bulk_tpm[common].sum(axis=1).values
    normR = TREF.loc[common].sum(axis=0).values
    B = bulk_tpm[sig].values / np.maximum(normB[:, None], 1e-12) * 1e6
    R = TREF.loc[sig].values / np.maximum(normR[None, :], 1e-12) * 1e6
    V = TREF_VAR.loc[sig].values / np.maximum(normR[None, :], 1e-12) * 1e6

    w = (R / (V + 1e-9)).sum(axis=1)
    pos = w[np.isfinite(w) & (w > 0)]
    if pos.size:
        w = np.nan_to_num(np.minimum(w, 100 * np.median(pos)), nan=1.0, posinf=1.0)
    w = np.maximum(w, 1e-6)

    # Row-scale the weighted system so the objective is O(1); SLSQP otherwise
    # stalls at the starting point (observed fun ~2e11, success=False).
    sw = np.sqrt(w)
    Rw = R * sw[:, None]
    n = R.shape[1]
    cons = [{"type": "ineq", "fun": lambda x: 1.0 - np.sum(x)}]
    bounds = [(0, None)] * n

    W = np.zeros((B.shape[0], n))
    nfail = 0
    from scipy.optimize import nnls
    for i in range(B.shape[0]):
        b = np.nan_to_num(B[i], nan=0.0)
        bw = b * sw
        # warm start from unconstrained NNLS, then enforce sum(x) <= 1
        try:
            x_nn, _ = nnls(Rw, bw)
        except Exception:
            x_nn = np.full(n, 1.0 / n)
        x0 = x_nn.copy()
        if x0.sum() > 1.0:
            x0 = x0 / x0.sum() * 0.999
        if x0.sum() <= 0:
            x0 = np.full(n, 1.0 / n)

        def f(x, bw=bw):
            r = Rw @ x - bw
            return float(np.sum(r * r))
        def g(x, bw=bw):
            r = Rw @ x - bw
            return 2.0 * Rw.T @ r

        res = minimize(f, x0, jac=g, bounds=bounds, constraints=cons,
                       method="SLSQP", options={"maxiter": 1000, "ftol": 1e-12})
        if not res.success or res.fun > f(x0) * 1.001:
            nfail += 1
            W[i] = np.clip(x0, 0, None)
        else:
            W[i] = np.clip(res.x, 0, None)
    if nfail:
        print(f"    EPIC: {nfail}/{B.shape[0]} fell back to NNLS warm start")

    prop = pd.DataFrame(W, index=bulk_tpm.index, columns=TREF.columns)
    prop["otherCells"] = np.clip(1 - prop.sum(axis=1), 0, 1)

    # mRNA -> cell fractions
    tinds = [MRNA_CELL.index.get_loc(c) if c in MRNA_CELL.index else MRNA_CELL.index.get_loc("default")
             for c in prop.columns]
    mf = MRNA_CELL.values[tinds]
    cf = prop.values / mf[None, :]
    cf = cf / cf.sum(axis=1, keepdims=True)
    frac = pd.DataFrame(cf, index=prop.index, columns=prop.columns)
    return prop, frac


# ================================================================ quanTIseq
def run_quantiseq(bulk_tpm, is_tumor=True):
    """bulk_tpm: samples x genes linear TPM. Returns fractions (samples x 11 incl Other)."""
    sig = TIL10.copy()
    drop = set()
    if is_tumor:
        drop |= ABGENES
    sig = sig.loc[[g for g in sig.index if g not in drop]]
    genes = [g for g in sig.index if g in bulk_tpm.columns]
    print(f"    quanTIseq: signature genes present {len(genes)}/{sig.shape[0]}")
    A0 = np.nan_to_num(sig.loc[genes].values, nan=0.0)
    B0 = np.nan_to_num(bulk_tpm[genes].values, nan=0.0)
    # Scale-match the bulk to the signature: bring each sample to the same total
    # over the shared signature genes as an average reference column.  Without
    # this a 257-gene TCGA subset sits on a different scale and the simplex
    # constraint forces "Other" to zero.
    target = float(A0.sum(axis=0).mean())
    B0 = B0 / np.maximum(B0.sum(axis=1, keepdims=True), 1e-12) * target

    n = A0.shape[1]
    sc = np.linalg.norm(A0, 2)
    A = A0 / sc
    Bm = B0 / sc
    x0 = np.full(n, 1.0 / n)
    cons = [{"type": "ineq", "fun": lambda x: 1.0 - np.sum(x)}]
    bounds = [(0, None)] * n

    out = np.zeros((Bm.shape[0], n + 1))
    from scipy.optimize import nnls
    for i in range(Bm.shape[0]):
        b = np.nan_to_num(Bm[i], nan=0.0)
        try:
            x_nn, _ = nnls(A, b)
        except Exception:
            x_nn = np.full(n, 1.0 / n)
        x0 = x_nn.copy()
        if x0.sum() > 1.0:
            x0 = x0 / x0.sum() * 0.999
        if x0.sum() <= 0:
            x0 = np.full(n, 1.0 / n)
        res = minimize(lambda x, b=b: float(np.sum((A @ x - b) ** 2)),
                       x0, jac=lambda x, b=b: 2.0 * A.T @ (A @ x - b),
                       bounds=bounds, constraints=cons, method="SLSQP",
                       options={"maxiter": 1000, "ftol": 1e-12})
        est = np.clip(res.x if res.success else x0, 0, None)
        est_sum = est.sum()
        est = est / TIL_SCALING
        if est.sum() > 0:
            est = est / est.sum() * est_sum
        out[i, :n] = est
        out[i, n] = max(0.0, 1.0 - est.sum())

    frac = pd.DataFrame(out, index=bulk_tpm.index, columns=list(TIL10.columns) + ["Other"])
    frac = frac.div(frac.sum(axis=1), axis=0)
    return frac


# ================================================================ DEPTOR split
EPIC_MAP = {"Bcells": "Bcell", "CAFs": "Stroma", "CD4_Tcells": "Tcell", "CD8_Tcells": "Tcell",
            "Endothelial": "Endo", "Macrophages": "Myeloid", "NKcells": "NK", "otherCells": "Malignant"}
QS_MAP = {"B.cells": "Bcell", "Macrophages.M1": "Myeloid", "Macrophages.M2": "Myeloid",
          "Monocytes": "Myeloid", "Neutrophils": "Myeloid", "NK.cells": "NK",
          "T.cells.CD4": "Tcell", "T.cells.CD8": "Tcell", "Tregs": "Tcell",
          "Dendritic.cells": "Myeloid", "Other": "Malignant"}
IMMUNE = ["Tcell", "NK", "Bcell", "Myeloid"]


def split_depTOR(frac, mapping, dep_per_comp):
    comp = pd.DataFrame(0.0, index=frac.index, columns=list(dep_per_comp.index))
    for col, c in mapping.items():
        if col in frac.columns and c in comp.columns:
            comp[c] = comp[c] + frac[col].values
    contrib = comp.mul(dep_per_comp, axis=1)
    tot = contrib.sum(axis=1).replace(0, np.nan)
    out = pd.DataFrame(index=frac.index)
    out["DEPTOR_tumor_frac"] = contrib["Malignant"] / tot
    out["DEPTOR_immune_frac"] = contrib[IMMUNE].sum(axis=1) / tot
    out["DEPTOR_myeloid_frac"] = contrib["Myeloid"] / tot
    out["DEPTOR_stroma_frac"] = contrib["Stroma"] / tot if "Stroma" in contrib else np.nan
    for c in comp.columns:
        out[f"prop_{c}"] = comp[c]
    out["prop_Immune"] = comp[IMMUNE].sum(axis=1)
    return out, comp


# ================================================================ run
if __name__ == "__main__":
    COHORTS = {"TCGA": False, "GSE65858": True, "GSE41613": True}
    summary, conc_rows = [], []

    for name, is_array in COHORTS.items():
        print(f"\n===== {name} =====")
        expr = load_cohort(name)
        print(f"  raw: {expr.shape}, range {np.nanmin(expr.values):.2f}-{np.nanmax(expr.values):.2f}")
        tpm = to_linear_tpm(expr, is_array)
        print(f"  TPM: {tpm.shape}")

        prop, efrac = run_epic(tpm)
        qfrac = run_quantiseq(tpm, is_tumor=True)

        efrac.to_csv(os.path.join(OUT, f"{name}_EPIC_fractions.csv"))
        qfrac.to_csv(os.path.join(OUT, f"{name}_quantiseq_fractions.csv"))

        esplit, ecomp = split_depTOR(efrac, EPIC_MAP, DEPTOR_C)
        qsplit, qcomp = split_depTOR(qfrac, QS_MAP, DEPTOR_C)
        esplit.to_csv(os.path.join(OUT, f"{name}_EPIC_DEPTOR_split.csv"))
        qsplit.to_csv(os.path.join(OUT, f"{name}_quantiseq_DEPTOR_split.csv"))

        print("  EPIC  mean fractions:", efrac.mean().round(3).to_dict())
        print("  qTseq mean fractions:", qfrac.mean().round(3).to_dict())
        print(f"  EPIC  tumor/immune DEPTOR: {esplit.DEPTOR_tumor_frac.mean():.3f} / {esplit.DEPTOR_immune_frac.mean():.3f}")
        print(f"  qTseq tumor/immune DEPTOR: {qsplit.DEPTOR_tumor_frac.mean():.3f} / {qsplit.DEPTOR_immune_frac.mean():.3f}")

        summary.append(dict(cohort=name, n=len(tpm),
                            EPIC_other=round(float(efrac.otherCells.mean()), 3),
                            EPIC_immune=round(float(efrac[[c for c in efrac.columns if c != 'otherCells']].sum(axis=1).mean()), 3),
                            qT_Other=round(float(qfrac.Other.mean()), 3),
                            qT_immune=round(float(1 - qfrac.Other.mean()), 3),
                            EPIC_tumor_DEPTOR=round(float(esplit.DEPTOR_tumor_frac.mean()), 3),
                            EPIC_immune_DEPTOR=round(float(esplit.DEPTOR_immune_frac.mean()), 3),
                            qT_tumor_DEPTOR=round(float(qsplit.DEPTOR_tumor_frac.mean()), 3),
                            qT_immune_DEPTOR=round(float(qsplit.DEPTOR_immune_frac.mean()), 3)))

        # concordance between the two methods + with the old custom solver
        common = efrac.index.intersection(qfrac.index)
        conc_rows.append(dict(
            cohort=name, comparison="EPIC_other_vs_qT_Other",
            r=round(float(pearsonr(efrac.loc[common, "otherCells"], qfrac.loc[common, "Other"])[0]), 3),
            rho=round(float(spearmanr(efrac.loc[common, "otherCells"], qfrac.loc[common, "Other"])[0]), 3)))
        conc_rows.append(dict(
            cohort=name, comparison="EPIC_immune_vs_qT_immune",
            r=round(float(pearsonr(1 - efrac.loc[common, "otherCells"], 1 - qfrac.loc[common, "Other"])[0]), 3),
            rho=round(float(spearmanr(1 - efrac.loc[common, "otherCells"], 1 - qfrac.loc[common, "Other"])[0]), 3)))
        conc_rows.append(dict(
            cohort=name, comparison="EPIC_tumorDEPTOR_vs_qT_tumorDEPTOR",
            r=round(float(pearsonr(esplit.loc[common, "DEPTOR_tumor_frac"], qsplit.loc[common, "DEPTOR_tumor_frac"])[0]), 3),
            rho=round(float(spearmanr(esplit.loc[common, "DEPTOR_tumor_frac"], qsplit.loc[common, "DEPTOR_tumor_frac"])[0]), 3)))

    pd.DataFrame(summary).to_csv(os.path.join(OUT, "deconv_summary.csv"), index=False)
    pd.DataFrame(conc_rows).to_csv(os.path.join(OUT, "validation_concordance.csv"), index=False)
    print("\n=== summary ===")
    print(pd.DataFrame(summary).to_string(index=False))
    print("\n=== concordance ===")
    print(pd.DataFrame(conc_rows).to_string(index=False))
    print(f"\nDONE -> {OUT}")
