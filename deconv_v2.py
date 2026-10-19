#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Deconvolution v2 -- fix the structural degeneracy of the previous NNLS run.

WHY THE OLD ONE FAILED (mathematically, not numerically)
  Old: H (ref) divided per-gene by its mean across cell types -> every column of
       H has mean exactly 1. Y (bulk) divided per-gene by its mean across samples
       -> every row of Y is ~1.  Therefore w = (1/7,...,1/7) reproduces Y
       EXACTLY.  Uniform is the exact optimum.  No solver can escape that.
  Fix: never divide by the per-gene mean.  Work on a linear abundance scale and
       normalise per SAMPLE (library-size style), so a gene keeps its biological
       magnitude across cell types.

METHOD
  reference : mean log1p(CP10K) per cell type -> expm1 -> CP10K -> /sum (composition)
  bulk      : log2 array -> 2**x -> floor -> /sum (composition)
  genes     : shared + cell-type specific (max/mean > spec_thr) + expressed
  solver    : NNLS on the simplex (non-negative, sum to 1)

VALIDATION (this is what the old run never did)
  - mean malignant fraction must land in 0.15-0.85
  - estimated malignant  vs independent keratin purity score   (expect r > 0.5)
  - estimated immune     vs independent immune score           (expect r > 0.5)
  - pseudo-bulk recovery: mix the reference at known fractions, check recovery
"""
import os, sys, gzip
import numpy as np, pandas as pd
from scipy.optimize import nnls
from scipy.stats import spearmanr, pearsonr

BASE = "WorkBuddy/2026-10-17-11-15-46"
REF = os.path.join(BASE, "data", "gse164690_centroid_full.csv")

REF_MK = {
    "Malignant": ["EPCAM","KRT5","KRT14","KRT6A","KRT17","KRT8","KRT18","KRT19","SFN","TP63","KRT13"],
    "Stroma":    ["COL1A1","COL1A2","COL3A1","DCN","LUM","FAP","PDGFRA","PDGFRB","ACTA2","MMP2"],
    "Endo":      ["PECAM1","CLDN5","VWF","CDH5","PLVAP","ENG"],
    "Tcell":     ["CD3D","CD3E","CD3G","CD2","CD8A","CD4","IL7R","TRAC","CD28"],
    "NK":        ["NKG7","GNLY","KLRF1","KLRD1","PRF1","NCAM1","KLRK1","XCL1","XCL2","FCGR3A"],
    "Bcell":     ["CD79A","CD79B","MS4A1","JCHAIN","MZB1","CD19","IGHG1"],
    "Myeloid":   ["LYZ","CD68","CD163","CSF1R","FCGR1A","ITGAM","VSIG4","MRC1","CD14","SPI1","AIF1","LST1","TYROBP"],
}


def load_reference(genes=None):
    ref = pd.read_csv(REF, index_col=0)
    if genes is not None:
        ref = ref[[g for g in genes if g in ref.columns]]
    Rl = np.expm1(ref.values.astype(float))          # -> CP10K-like linear
    Rl = np.clip(Rl, 0, None)
    return ref.index.tolist(), ref.columns.tolist(), Rl


def select_genes(ref, expr_index, spec_thr=2.0, min_lin=1.0, top_n=None, per_class=None):
    """cell-type specific + reasonably expressed genes.

    per_class : if set, take the top `per_class` genes FOR EACH CELL TYPE (by
                specificity x expression) and union them.  This is essential --
                a global top-N ranking is dominated by keratins/collagens and the
                immune compartments end up with no markers at all (-> fraction 0).
    """
    Rl = np.expm1(ref.values.astype(float))
    Rl = np.clip(Rl, 0, None)
    mx = Rl.max(axis=0); mn = Rl.mean(axis=0)
    spec = mx / np.maximum(mn, 1e-9)
    present = np.array([g in expr_index for g in ref.columns])

    if per_class is not None:
        keep = set()
        for c in ref.index:
            v = Rl[ref.index.get_loc(c)]
            oth = np.delete(Rl, ref.index.get_loc(c), axis=0).max(axis=0)
            sc = (v - oth) / (v + oth + 1e-6)          # relative specificity per class
            ok = present & (v >= min_lin) & (sc > 0)
            idx = np.where(ok)[0]
            if len(idx) == 0: continue
            order = idx[np.argsort(-sc[idx])][:per_class]
            keep |= {ref.columns[i] for i in order}
        return sorted(keep)

    ok = (spec >= spec_thr) & (mx >= min_lin) & present
    genes = [g for g, k in zip(ref.columns, ok) if k]
    if top_n is not None and len(genes) > top_n:
        sc = dict(zip(ref.columns, spec * np.log1p(mx)))
        genes = sorted(genes, key=lambda g: -sc[g])[:top_n]
    return genes


def deconvolve(Y_log2, genes, ref, spec_thr=2.0, min_lin=1.0, top_n=None,
               per_class=None, per_gene_scale=None, floor=0.0, weight=None, verbose=True):
    """
    Y_log2 : samples x all-genes DataFrame (already log2, gene symbols as columns)
    weight : None -> plain NNLS (dominated by high-magnitude genes)
             'norm' -> divide gene g of BOTH H and Y by ||H[:,g]||, so every gene
                       contributes equally to the objective.  This is a standard
                       column scaling of the design matrix; unlike dividing by the
                       per-gene mean it does NOT make the uniform vector an exact
                       solution, so it rescues the immune compartments without
                       reintroducing the degeneracy.
    returns fractions DataFrame (samples x cell types) summing to 1
    """
    g = select_genes(ref, Y_log2.columns, spec_thr, min_lin, top_n, per_class)
    if verbose:
        print(f"    genes used: {len(g)}")
    H = np.clip(np.expm1(ref[g].values.astype(float)), 0, None)     # cell types x genes
    B = np.clip(2.0 ** Y_log2[g].values.astype(float) - floor, 0, None)  # samples x genes

    if per_gene_scale == "refmax":      # divide each gene by its reference max
        s = np.maximum(H.max(axis=0), 1e-9)
        H = H / s; B = B / s
    # normalise per sample / per cell-type profile to composition over used genes
    B = B / np.maximum(B.sum(axis=1, keepdims=True), 1e-9)
    H = H / np.maximum(H.sum(axis=1, keepdims=True), 1e-9)

    Hf, Bf = H, B
    if weight == "norm":
        cw = np.maximum(np.linalg.norm(H, axis=0), 1e-12)
        Hf = H / cw[None, :]
        Bf = B / cw[None, :]

    W = np.zeros((B.shape[0], H.shape[0]))
    for i in range(Bf.shape[0]):
        w, _ = nnls(Hf.T, Bf[i])
        W[i] = w / w.sum() if w.sum() > 0 else np.full(H.shape[0], 1.0 / H.shape[0])
    return pd.DataFrame(W, columns=ref.index, index=Y_log2.index), len(g)


def deconvolve_als(Y_log2, ref, spec_thr=2.0, min_lin=1.0, top_n=None, per_class=None,
                   n_iter=8, verbose=True):
    """
    Cross-platform-aware deconvolution.
    Model:  log2(bulk_sg) = a_g + log2( sum_c f_sc * h_cg )
      a_g : per-gene offset (probe affinity / platform effect) -- nuisance, fitted
      f_sc: fractions (simplex)
    Fitted by alternating least squares, so an array-vs-scRNA scale mismatch no
    longer collapses the solution to the uniform vector.
    """
    g = select_genes(ref, Y_log2.columns, spec_thr, min_lin, top_n, per_class)
    if verbose:
        print(f"    genes used: {len(g)}")
    H = np.clip(np.expm1(ref[g].values.astype(float)), 0, None)   # C x G
    H = np.maximum(H, 1e-6)
    Y = Y_log2[g].values.astype(float)                            # S x G (log2)
    S, C = Y.shape[0], H.shape[0]

    f = np.full((S, C), 1.0 / C)
    a = np.zeros(Y.shape[1])
    for it in range(n_iter):
        # --- given a, solve fractions on the linear scale
        T = np.clip(2.0 ** (Y - a[None, :]), 0, None)
        for i in range(S):
            w, _ = nnls(H.T, T[i])
            f[i] = w / w.sum() if w.sum() > 0 else 1.0 / C
        # --- given f, update per-gene offsets
        pred = np.log2(np.maximum(f @ H, 1e-9))
        a = (Y - pred).mean(axis=0)
    resid = Y - (a[None, :] + np.log2(np.maximum(f @ H, 1e-9)))
    if verbose:
        print(f"    ALS residual SD (log2): {resid.std():.3f}")
    return pd.DataFrame(f, columns=ref.index, index=Y_log2.index), len(g), resid.std()


def pseudobulk_with_probe_effects(ref, n=300, seed=0, alpha=0.6, offset_sd=1.5):
    """known fractions + random per-gene log2 offsets (simulating array probes)"""
    rng = np.random.default_rng(seed)
    H = np.clip(np.expm1(ref.values.astype(float)), 0, None)
    Hn = H / np.maximum(H.sum(axis=1, keepdims=True), 1e-9)
    truth = rng.dirichlet(np.full(Hn.shape[0], alpha), size=n)
    mix = truth @ Hn
    obs = np.log2(np.clip(mix, 1e-7, None)) + rng.normal(0, offset_sd, mix.shape[1])[None, :]
    obs = obs + rng.normal(0, 0.08, obs.shape)
    return pd.DataFrame(obs, columns=ref.columns, index=[f"syn{i}" for i in range(n)]), truth


def pseudobulk_recovery(ref, n=300, seed=0, frac_alpha=0.6, **kw):
    """mix reference profiles at known Dirichlet fractions -> log2 'array-like' -> deconvolve"""
    rng = np.random.default_rng(seed)
    H = np.clip(np.expm1(ref.values.astype(float)), 0, None)
    Hn = H / np.maximum(H.sum(axis=1, keepdims=True), 1e-9)
    truth = rng.dirichlet(np.full(Hn.shape[0], frac_alpha), size=n)
    mix = truth @ Hn                                     # n x genes, linear composition
    # imitate an array: log2 with a floor and a bit of multiplicative noise
    log2mix = np.log2(np.clip(mix, 1e-6, None) + 2.0 ** -6)
    df = pd.DataFrame(log2mix + rng.normal(0, 0.08, log2mix.shape),
                      columns=ref.columns, index=[f"syn{i}" for i in range(n)])
    est, ng = deconvolve(df, None, ref, verbose=False, **kw)
    err = np.abs(est.values - truth)
    print(f"  pseudo-bulk recovery ({ng} genes): mean|err|={err.mean():.3f}  "
          f"max|err|={err.max():.3f}  r(Malignant)={pearsonr(est['Malignant'], truth[:,0])[0]:.3f}")
    return est, truth, err.mean()


def score(df, mks):
    zs = []
    for g in mks:
        if g in df.columns:
            v = df[g].astype(float)
            if v.std() > 0:
                zs.append((v - v.mean()) / v.std())
    return pd.concat(zs, axis=1).mean(axis=1) if zs else pd.Series(np.nan, index=df.index)


if __name__ == "__main__":
    print("reference:", REF)
    ref = pd.read_csv(REF, index_col=0)
    print("  ", ref.shape)

    cfgs = [dict(spec_thr=2.0, min_lin=1.0, top_n=500),
            dict(spec_thr=3.0, min_lin=1.0, top_n=300)]

    print("\n[0] solver sanity on clean pseudo-bulk (no probe effects)")
    for kw in cfgs:
        print("   cfg", kw)
        pseudobulk_recovery(ref, n=200, **kw)

    print("\n[1] THE REAL TEST -- pseudo-bulk with random per-gene probe offsets")
    print("    (simulates array-vs-scRNA scale mismatch that killed the old NNLS)")
    for kw in cfgs:
        df, truth = pseudobulk_with_probe_effects(ref, n=300, offset_sd=1.5)
        for name, fn in [("naive compositional", lambda: deconvolve(df, None, ref, verbose=False, **kw)[0]),
                         ("ALS (per-gene offset)", lambda: deconvolve_als(df, ref, verbose=False, **kw)[0])]:
            est = fn()
            err = np.abs(est.values - truth)
            rm = pearsonr(est["Malignant"], truth[:, 0])[0]
            print(f"    {kw['top_n']:>4}g  {name:22s} runif={est.iloc[0].round(2).tolist()} "
                  f"mean|err|={err.mean():.3f} r(Mal)={rm:.3f}")
