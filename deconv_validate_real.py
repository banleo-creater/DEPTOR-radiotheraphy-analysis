#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Validate deconv_v2 on a real array cohort (GSE65858) against independent scores."""
import os, sys, gzip
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from deconv_v2 import deconvolve, load_reference, REF_MK, score

BASE = "WorkBuddy/2026-10-17-11-15-46"
GEO = "<PATH>"


def gpl_symbols(soft_gz, sym_cands=("symbol", "ilmn_gene", "gene symbol", "gene_symbol")):
    rows = {}
    with gzip.open(soft_gz, "rt", errors="ignore") as f:
        intab = False; sym_col = None
        for line in f:
            if line.startswith("!platform_table_begin"): intab = True; continue
            if line.startswith("!platform_table_end"): break
            if not intab: continue
            p = line.rstrip("\n").split("\t")
            if sym_col is None:
                cols = [c.strip().lower() for c in p]
                for cand in sym_cands:
                    if cand in cols: sym_col = cols.index(cand); break
                continue
            if sym_col is None or len(p) <= sym_col: continue
            pid = p[0].strip(); sym = p[sym_col].strip().strip('"')
            if pid and sym and sym not in ("", "---"):
                rows[pid] = sym.split("///")[0].strip()
    return rows


def parse_matrix(sm_gz):
    with gzip.open(sm_gz, "rt", errors="ignore") as f:
        lines = f.readlines()
    s = next(i for i, l in enumerate(lines) if l.startswith("!series_matrix_table_begin"))
    e = next(i for i, l in enumerate(lines) if l.startswith("!series_matrix_table_end"))
    hdr = [h.strip().strip('"') for h in lines[s + 1].rstrip("\n").split("\t")]
    rows = []
    for l in lines[s + 2:e]:
        p = l.rstrip("\n").split("\t")
        rows.append([p[0].strip().strip('"')] +
                    [float(x) if x.strip() not in ("", "NA", "null", "NaN") else np.nan for x in p[1:]])
    return pd.DataFrame(rows, columns=["ID"] + hdr[1:]).set_index("ID")


def main():
    print("loading GSE65858 ...")
    p2s = gpl_symbols(os.path.join(BASE, "data", "plat", "GPL10558.soft.gz"))
    expr = parse_matrix(os.path.join(GEO, "GSE65858", "GSE65858_series_matrix.txt.gz"))
    expr["symbol"] = [p2s.get(i) for i in expr.index]
    expr = expr.dropna(subset=["symbol"]).replace({"symbol": {"DEPDC6": "DEPTOR"}})
    expr = expr.groupby("symbol").max().T                     # samples x genes, log2
    print("  ", expr.shape)

    ref = pd.read_csv(os.path.join(BASE, "data", "gse164690_centroid_full.csv"), index_col=0)

    for cfg in [dict(spec_thr=2.0, min_lin=1.0, top_n=500),
                dict(spec_thr=3.0, min_lin=1.0, top_n=300)]:
        print("\n=== cfg", cfg, "===")
        frac, ng = deconvolve(expr, None, ref, verbose=True, **cfg)
        print("  mean fractions:", frac.mean().round(3).to_dict())
        print("  SD  fractions:", frac.std().round(3).to_dict())
        purity = score(expr, REF_MK["Malignant"])
        immune = score(expr, sum([REF_MK[k] for k in ("Tcell", "NK", "Bcell", "Myeloid")], []))
        stroma = score(expr, REF_MK["Stroma"])
        for a, b, name in [(frac["Malignant"], purity, "malignant vs keratin"),
                           (frac[["Tcell", "NK", "Bcell", "Myeloid"]].sum(axis=1), immune, "immune vs immune-score"),
                           (frac["Stroma"], stroma, "stroma vs stroma-score")]:
            r, p = spearmanr(a, b)
            print(f"  {name:26s} rho={r:+.3f} p={p:.2g}")



if __name__ == '__main__':
    main()
