#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
In-silico ("virtual knockout") analysis of DEPTOR and radiosensitivity
in TCGA-HNSC (PanCancer Atlas 2018).

Virtual knockout proxy
----------------------
DEPTOR is a negative regulator of mTORC1. Its knockout is approximated by the
DEPTOR-LOW end of the expression spectrum (bottom tertile). We compare
"DEPTOR-low (virtual KO)" vs "DEPTOR-high (intact)" on pathway scores that
determine intrinsic radiosensitivity:

    DDR (DNA damage repair)   higher -> radioRESISTANT
    PROLIF (cell cycle)       higher -> generally more radiosensitive (cycling)
    APOP - ANTI_APOP          higher -> more apoptosis-prone -> radiosensitive
    HYPOXIA                   higher -> radioRESISTANT
    MTORC1 targets (RPS6KB1/EIF4EBP1)  higher -> growth-factor driven

All scores are mean z-scores of their gene sets. Analyses are stratified by
RADIATION_THERAPY and by HPV (SUBTYPE).
"""
import os
import numpy as np, pandas as pd
from scipy.stats import spearmanr, mannwhitneyu
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

BASE = "WorkBuddy/2026-10-17-11-15-46"
OUT = os.path.join(BASE, "data", "results_vko"); os.makedirs(OUT, exist_ok=True)

SETS = {
 "DDR": ["ATM","ATR","BRCA1","BRCA2","RAD51","RAD51B","RAD51C","RAD51D","CHEK1","CHEK2",
         "MDC1","MRE11","NBN","PRKDC","XRCC4","XRCC5","XRCC6","PARP1","FANCD2","FANCA",
         "PALB2","BRIP1","MLH1","MSH2","MSH6","PMS2"],
 "PROLIF": ["CDK1","CDK2","CDK4","CDK6","CCNB1","CCNE1","CCNA2","PLK1","AURKA","AURKB",
            "WEE1","MKI67","PCNA","TOP2A","BUB1","BUB1B","MAD2L1"],
 "APOP": ["BAX","BAK1","CASP3","CASP8","CASP9","TP53","BBC3","BID","PMAIP1","APAF1"],
 "ANTI_APOP": ["BCL2","BCL2L1","MCL1","XIAP","BIRC5"],
 "HYPOXIA": ["HIF1A","VEGFA","CA9","SLC2A1","LDHA","PGK1","ADM"],
 "MTORC1": ["RPS6KB1","EIF4EBP1"],     # downstream readouts (DEPTOR excluded)
}

clin = pd.read_csv(os.path.join(BASE, "data", "tcga_hnsc_pca_deptor.csv"))
genes = pd.read_csv(os.path.join(BASE, "data", "tcga_hnsc_vko_genes.csv"))
df = clin.merge(genes, on="sampleId", how="inner")
print("merged samples:", len(df))

# z-score each gene across samples
for g in set(sum(SETS.values(), [])):
    if g in df.columns:
        v = pd.to_numeric(df[g], errors="coerce")
        df[g] = v
        mu, sd = v.mean(), v.std()
        df[g + "_z"] = (v - mu) / sd if sd and not np.isnan(sd) else np.nan

def score(name):
    gs = [g + "_z" for g in SETS[name] if g + "_z" in df.columns]
    return df[gs].mean(axis=1)

for s in SETS:
    df[f"score_{s}"] = score(s)
df["score_APOP_NET"] = df["score_APOP"] - df["score_ANTI_APOP"]

df["DEPTOR"] = pd.to_numeric(df["DEPTOR_RSEM"], errors="coerce")
df["RT"] = df["RADIATION_THERAPY"].astype(str).str.strip()
df["HPV"] = np.where(df["SUBTYPE"].astype(str).str.contains("HPV-"), "HPV-",
            np.where(df["SUBTYPE"].astype(str).str.contains("HPV\\+"), "HPV+", "NA"))

SCORES = ["score_DDR","score_PROLIF","score_APOP_NET","score_HYPOXIA","score_MTORC1"]

# ---------- 1) DEPTOR vs pathway scores ----------
print("\n=== Spearman: DEPTOR vs pathway scores ===")
rows = []
strata = [("All", df)] + [(f"RT={v}", df[df.RT == v]) for v in ["Yes","No"]] \
       + [(f"RT=Yes/{h}", df[(df.RT == "Yes") & (df.HPV == h)]) for h in ["HPV+","HPV-"]]
for sname, sub in strata:
    for sc in SCORES:
        s = sub[["DEPTOR", sc]].dropna()
        if len(s) < 10: continue
        rho, p = spearmanr(s.iloc[:,0], s.iloc[:,1])
        rows.append(dict(stratum=sname, score=sc.replace("score_",""), n=len(s),
                         rho=round(rho,3), p=round(p,4)))
tab = pd.DataFrame(rows)
tab.to_csv(os.path.join(OUT, "vko_spearman.csv"), index=False)
piv = tab.pivot(index="score", columns="stratum", values="rho")
print(piv.to_string())

# ---------- 2) DEPTOR-low (virtual KO) vs DEPTOR-high ----------
print("\n=== Virtual KO: DEPTOR-low (bottom tertile) vs DEPTOR-high (top tertile) ===")
d = df.dropna(subset=["DEPTOR"]).copy()
q1, q2 = d.DEPTOR.quantile([1/3, 2/3])
d["KO"] = np.where(d.DEPTOR <= q1, "DEPTOR_low(vKO)",
          np.where(d.DEPTOR >= q2, "DEPTOR_high", "mid"))
res = []
for sname, sub in [("All", d)] + [(f"RT={v}", d[d.RT == v]) for v in ["Yes","No"]] \
                 + [(f"RT=Yes/{h}", d[(d.RT=="Yes") & (d.HPV==h)]) for h in ["HPV+","HPV-"]]:
    lo = sub[sub.KO == "DEPTOR_low(vKO)"]; hi = sub[sub.KO == "DEPTOR_high"]
    if len(lo) < 8 or len(hi) < 8: continue
    for sc in SCORES:
        a = lo[sc].dropna(); b = hi[sc].dropna()
        if len(a) < 8 or len(b) < 8: continue
        u, p = mannwhitneyu(a, b)
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        dcoh = (a.mean() - b.mean()) / pooled if pooled > 0 else np.nan
        res.append(dict(stratum=sname, score=sc.replace("score_",""),
                        n_KO=len(a), n_high=len(b),
                        mean_KO=round(a.mean(),3), mean_high=round(b.mean(),3),
                        diff_KO_minus_high=round(a.mean()-b.mean(),3),
                        cohens_d=round(dcoh,3), p=round(p,4)))
res = pd.DataFrame(res)
res.to_csv(os.path.join(OUT, "vko_KO_vs_high.csv"), index=False)
print(res[res.stratum.isin(["All","RT=Yes"])].to_string(index=False))

# ---------- 3) figures ----------
# (a) KO vs high bars in RT=Yes
sub = res[res.stratum == "RT=Yes"]
if len(sub):
    fig, ax = plt.subplots(figsize=(8,4))
    x = np.arange(len(sub)); w = 0.38
    ax.bar(x-w/2, sub.mean_KO, w, label="DEPTOR-low (virtual KO)", color="#d62728")
    ax.bar(x+w/2, sub.mean_high, w, label="DEPTOR-high", color="#1f77b4")
    for i, r_ in enumerate(sub.itertuples()):
        ax.text(i, max(r_.mean_KO, r_.mean_high)+0.05, f"p={r_.p:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(sub.score, fontsize=9)
    ax.axhline(0, color="k", lw=.7)
    ax.set_ylabel("pathway score (mean z)"); ax.set_title("Virtual knockout of DEPTOR in RT-treated HNSCC\n(TCGA-HNSC, bottom vs top tertile)")
    ax.legend(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "vko_KO_vs_high_RTyes.png"), dpi=140); plt.close()

# (b) DEPTOR vs DDR/PROLIF/HYPOXIA scatter (RT=Yes)
sub = df[df.RT == "Yes"].dropna(subset=["DEPTOR"])
fig, axes = plt.subplots(1, 3, figsize=(13,4))
for ax, sc in zip(axes, ["score_DDR","score_PROLIF","score_HYPOXIA"]):
    ax.scatter(sub["DEPTOR"], sub[sc], s=14, alpha=.55, c=np.where(sub.HPV=="HPV-","#d62728","#1f77b4"))
    ss = sub[["DEPTOR", sc]].dropna()
    if len(ss) > 5:
        rho, p = spearmanr(ss.iloc[:,0], ss.iloc[:,1])
        ax.set_title(f"{sc.replace('score_','')}\nrho={rho:+.2f} (p={p:.2g})", fontsize=9)
    ax.set_xlabel("DEPTOR (log2 RSEM)"); ax.set_ylabel(sc.replace("score_","")+" (z)")
fig.suptitle("DEPTOR vs radiosensitivity pathways in RT-treated HNSCC (red=HPV−, blue=HPV+)", fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "vko_DEPTOR_vs_pathways_RTyes.png"), dpi=140); plt.close()

# (c) heatmap of rho across strata
if len(tab):
    h = tab.pivot(index="score", columns="stratum", values="rho")
    order = ["All","RT=Yes","RT=No","RT=Yes/HPV-","RT=Yes/HPV+"]
    h = h[[c for c in order if c in h.columns]]
    fig, ax = plt.subplots(figsize=(6.5,3.6))
    im = ax.imshow(h.values, cmap="RdBu_r", vmin=-.4, vmax=.4, aspect="auto")
    ax.set_xticks(range(h.shape[1])); ax.set_xticklabels(h.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(h.shape[0])); ax.set_yticklabels(h.index, fontsize=9)
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            if not np.isnan(h.values[i,j]):
                ax.text(j, i, f"{h.values[i,j]:+.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Spearman rho: DEPTOR vs pathway scores")
    fig.colorbar(im, ax=ax, fraction=.046); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "vko_rho_heatmap.png"), dpi=140); plt.close()

print("\nSaved to", OUT, "DONE.")
