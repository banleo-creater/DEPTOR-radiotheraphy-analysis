#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Figures for the three-cohort deconvolution / mediation analysis."""
import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore")

BASE = "WorkBuddy/2026-10-17-11-15-46"
OUT = os.path.join(BASE, "data", "results_pooled3"); os.makedirs(OUT, exist_ok=True)

TUMOR = ["KRT5", "KRT6A", "KRT14", "KRT17", "KRT13", "SFN", "TP63", "KRT19", "EPCAM"]
STROMA = ["COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "FAP", "PDGFRA", "PDGFRB", "ACTA2", "MMP2"]
IMMUNE = ["PTPRC", "CD3D", "CD3E", "CD2", "CD8A", "CD79A", "MS4A1", "CD68", "LYZ", "NKG7", "GNLY"]


def zscore_mean(df, genes):
    zs = []
    for g in genes:
        if g in df.columns:
            v = pd.to_numeric(df[g], errors="coerce")
            if v.std() > 0:
                zs.append((v - v.mean()) / v.std())
    return pd.concat(zs, axis=1).mean(axis=1) if zs else pd.Series(np.nan, index=df.index)


def load(name):
    if name == "TCGA-HNSC":
        a = pd.read_csv(os.path.join(BASE, "data", "results_deconv", "deconv_per_sample.csv"), index_col=0)
        d = pd.DataFrame({"DEPTOR": a["DEPTOR"], "hypoxia": a["Hypoxia_score"]})
        d = d.join(a[[g for g in TUMOR + STROMA + IMMUNE if g in a.columns]])
    elif name == "GSE65858":
        a = pd.read_csv(os.path.join(BASE, "data", "results_deconv65858", "gse65858_deconv_sample.csv"), index_col=0)
        d = pd.DataFrame({"DEPTOR": a["DEPTOR"], "hypoxia": a["hypoxia"]})
        import gzip, sys
        sys.path.insert(0, os.path.join(BASE, "scripts"))
        from deconv_validate_real import parse_matrix, gpl_symbols
        p2s = gpl_symbols(os.path.join(BASE, "data", "plat", "GPL10558.soft.gz"))
        e = parse_matrix("<PATH>")
        e["symbol"] = [p2s.get(i) for i in e.index]
        e = e.dropna(subset=["symbol"]).replace({"symbol": {"DEPDC6": "DEPTOR"}}).groupby("symbol").max().T
        d = d.join(e[[g for g in TUMOR + STROMA + IMMUNE if g in e.columns]])
    else:
        a = pd.read_csv(os.path.join(BASE, "data", "results_deconv41613", "GSE41613_deconv_sample.csv"), index_col=0)
        d = pd.DataFrame({"DEPTOR": a["DEPTOR"], "hypoxia": a["hypoxia"]})
        import sys
        sys.path.insert(0, os.path.join(BASE, "scripts"))
        from deconv_validate_real import parse_matrix
        import gzip
        p2s = {}
        with gzip.open(os.path.join(BASE, "data", "plat", "GPL570.annot.gz"), "rt", errors="ignore") as f:
            intab = False; sc = None
            for line in f:
                if line.startswith("!platform_table_begin"): intab = True; continue
                if line.startswith("!platform_table_end"): break
                if not intab: continue
                p = line.rstrip("\n").split("\t")
                if sc is None:
                    cols = [c.strip().lower() for c in p]
                    sc = cols.index("gene symbol") if "gene symbol" in cols else 2; continue
                if len(p) <= sc: continue
                pid = p[0].strip().strip('"'); sym = p[sc].strip().strip('"')
                if pid and sym and sym not in ("", "---"): p2s[pid] = sym.split("///")[0].strip()
        e = parse_matrix("<PATH>")
        e["symbol"] = [p2s.get(i) for i in e.index]
        e = e.dropna(subset=["symbol"]).groupby("symbol").max().T
        d = d.join(e[[g for g in TUMOR + STROMA + IMMUNE if g in e.columns]])
    d["tumour"] = zscore_mean(d, TUMOR)
    d["stroma"] = zscore_mean(d, STROMA)
    d["immune"] = zscore_mean(d, IMMUNE)
    return d.dropna(subset=["DEPTOR"])


print("loading cohorts ...")
D = {k: load(k) for k in ["TCGA-HNSC", "GSE65858", "GSE41613"]}

# ---- panel A: DEPTOR vs hypoxia (the robust link)
fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.3))
for ax, (name, d) in zip(axes, D.items()):
    dd = d.dropna(subset=["hypoxia"])
    r, p = spearmanr(dd.DEPTOR, dd.hypoxia)
    ax.scatter(dd.DEPTOR, dd.hypoxia, s=13, alpha=.55, color="#2c7fb8", edgecolor="none")
    b = np.polyfit(dd.DEPTOR, dd.hypoxia, 1)
    xs = np.linspace(dd.DEPTOR.min(), dd.DEPTOR.max(), 50)
    ax.plot(xs, np.polyval(b, xs), color="#d95f0e", lw=2)
    ax.set_title(f"{name}  (n={len(dd)})\n$\\rho$={r:+.2f}, p={p:.1e}", fontsize=9)
    ax.set_xlabel("DEPTOR expression (log2/RSEM)")
    ax.set_ylabel("hypoxia programme score")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "F_A_DEPTOR_hypoxia_3cohorts.png"), dpi=200); plt.close()

# ---- panel B: what does bulk DEPTOR track? (hypoxia / tumour / stroma / immune)
rows = []
for name, d in D.items():
    for v in ["hypoxia", "tumour", "stroma", "immune"]:
        dd = d.dropna(subset=[v])
        r, p = spearmanr(dd.DEPTOR, dd[v])
        rows.append(dict(cohort=name, variable=v, rho=r, p=p, n=len(dd)))
R = pd.DataFrame(rows)
R.to_csv(os.path.join(OUT, "F_B_association_matrix.csv"), index=False)

fig, ax = plt.subplots(figsize=(6.2, 3.4))
vars_ = ["hypoxia", "tumour", "stroma", "immune"]
x = np.arange(len(vars_)); w = 0.26
cols = {"TCGA-HNSC": "#2c7fb8", "GSE65858": "#41ab5d", "GSE41613": "#d95f0e"}
for i, name in enumerate(D):
    sub = R[R.cohort == name].set_index("variable").reindex(vars_)
    ax.bar(x + (i - 1) * w, sub.rho.values, w, label=name, color=cols[name], edgecolor="white")
    for xi, v, p in zip(x + (i - 1) * w, sub.rho.values, sub.p.values):
        off = 0.018 if v >= 0 else -0.018
        ax.text(xi, v + off, "*" if p < 0.05 else "", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=10, color="#222")
ax.axhline(0, color="#444", lw=1)
ax.set_ylim(-0.52, 0.44)
ax.set_xticks(x); ax.set_xticklabels(["hypoxia", "tumour content", "stroma", "immune"])
ax.set_ylabel("Spearman $\\rho$ with bulk DEPTOR")
ax.set_title("What bulk DEPTOR tracks (three cohorts)", fontsize=10)
ax.legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "F_B_what_DEPTOR_tracks.png"), dpi=200); plt.close()

# ---- panel C: mediation summary
med = pd.read_csv(os.path.join(OUT, "three_cohort_mediation.csv"))
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.2))
ax = axes[0]
ax.barh(np.arange(len(med))[::-1], med.a_rho.values, color="#2c7fb8")
ax.set_yticks(np.arange(len(med))[::-1]); ax.set_yticklabels(med.cohort, fontsize=9)
ax.axvline(0, color="#444", lw=1)
ax.set_xlim(-0.52, 0.06)
for i, (v, p) in enumerate(zip(med.a_rho.values, med.a_p.values)):
    ax.text(v - 0.015, len(med) - 1 - i, f"{v:+.2f}{'*' if p < 0.05 else ''}",
            va="center", ha="right", fontsize=8, color="#222")
ax.set_xlabel("a-path: DEPTOR → hypoxia ($\\rho$)")
ax.set_title("Replicated in 3/3 cohorts", fontsize=9)
ax = axes[1]
ax.barh(np.arange(len(med))[::-1], med.b_logHR.values, color="#d95f0e")
ax.set_yticks(np.arange(len(med))[::-1]); ax.set_yticklabels(med.cohort, fontsize=9)
ax.axvline(0, color="#444", lw=1)
for i, (v, p) in enumerate(zip(med.b_logHR.values, med.b_p.values)):
    ax.text(v + 0.006, len(med) - 1 - i, f"{v:+.2f}{'*' if p < 0.05 else ''}",
            va="center", ha="left", fontsize=8, color="#222")
ax.set_xlim(0, 0.29)
ax.set_xlabel("b-path: hypoxia → OS (log HR per SD)")
ax.set_title("Significant in the two larger cohorts", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "F_C_mediation_paths.png"), dpi=200); plt.close()

print(R.round(3).to_string(index=False))
print("\nSaved figures ->", OUT)
