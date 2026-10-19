"""Exploratory pivot: arbitrary gene(s) instead of DEPTOR in TCGA-HNSC.
Mirrors the DEPTOR pipeline: median-split survival, RT interaction Cox,
and correlation with deconvolved immune / myeloid / malignant fractions.
Usage: python gene_explore.py SSBP1 [SSBP1 ...]
Outputs -> data/results_gene/<GENE>/  (or shared folder for multiple)
"""
import os, sys, gzip, csv
import numpy as np
import pandas as pd
from scipy import stats
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GENES = sys.argv[1:] or ["SSBP1"]
ROOT = "data/results_gene"
os.makedirs(ROOT, exist_ok=True)
C_HPVP, C_HPVN = "#E8743B", "#3B6EA5"
GRID = "#DDDDDD"

# ---------- 1. pull genes from Xena full transcriptome ----------
zpath = "data/tcga_hnsc_hiseq.gz"
samples, gexpr = None, {}
with gzip.open(zpath, "rt") as f:
    r = csv.reader(f, delimiter="\t")
    hdr = next(r)
    samples = [s.replace("-01", "") for s in hdr[1:]]
    for row in r:
        if row[0] in GENES:
            gexpr[row[0]] = [float(v) for v in row[1:]]
gexpr = pd.DataFrame(gexpr, index=pd.Index(samples, name="sid"))
print("genes loaded:", list(gexpr.columns), "samples:", len(samples))

# ---------- 2. master clinical/deconv table ----------
df = pd.read_csv("data/results_deconv/deconv_per_sample.csv")
df["sid"] = df["sampleId"].str.replace("-01", "", regex=False)
df = df.merge(gexpr, on="sid", how="left")
df["RTb"] = (df["RT"] == "Yes").astype(int)
print("merged rows:", df.shape, "| SSBP1 nonnull:", df["SSBP1"].notna().sum())

def norm(x):
    return pd.to_numeric(x, errors="coerce")

def cox_hr(sub, gene, tcol, ecol):
    d = sub[[gene, tcol, ecol]].dropna()
    d = d[d[tcol] > 0].copy()
    if d[ecol].nunique() < 2 or d[gene].nunique() < 2:
        return None
    d["hi"] = (d[gene] >= d[gene].median()).astype(int)
    cf = CoxPHFitter()
    try:
        cf.fit(d.rename(columns={tcol: "T", ecol: "E"}), duration_col="T", event_col="E")
    except Exception:
        return None
    b = cf.summary.loc["hi"]
    return dict(HR=np.exp(b["coef"]), lo=np.exp(b["coef"] - 1.96 * b["se(coef)"]),
               hi=np.exp(b["coef"] + 1.96 * b["se(coef)"]), p=b["p"])

def interact(sub, gene, tcol, ecol):
    d = sub[[gene, "RTb", tcol, ecol]].dropna()
    d = d[d[tcol] > 0].copy()
    if len(d) < 30:
        return None
    d["hi"] = (d[gene] >= d[gene].median()).astype(int)
    d = d.rename(columns={"RTb": "RT"})
    cf = CoxPHFitter()
    try:
        cf.fit(d.rename(columns={tcol: "T", ecol: "E"}), duration_col="T", event_col="E",
               formula="hi * RT")
    except Exception:
        return None
    key = [k for k in cf.summary.index if "*" in k or ":" in k]
    if not key:
        return None
    b = cf.summary.loc[key[0]]
    return dict(HR=np.exp(b["coef"]), lo=np.exp(b["coef"] - 1.96 * b["se(coef)"]),
               hi=np.exp(b["coef"] + 1.96 * b["se(coef)"]), p=b["p"])

def km_panel(fig, spec, sub, gene, tcol, ecol, title):
    ax = fig.add_subplot(spec)
    d = sub[[gene, tcol, ecol]].dropna()
    d = d[d[tcol] > 0].copy()
    d["grp"] = (d[gene] >= d[gene].median()).map({True: f"{gene}-high", False: f"{gene}-low"})
    cols = {f"{gene}-high": C_HPVN, f"{gene}-low": C_HPVP}
    for g in [f"{gene}-high", f"{gene}-low"]:
        s = d[d.grp == g]
        if s.empty:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(s[tcol] / 12.0, event_observed=s[ecol].astype(bool), label=g)
        ax.plot(kmf.survival_function_, color=cols[g], lw=1.8, label=g)
        ax.fill_between(kmf.survival_function_.index,
                        kmf.confidence_interval_.iloc[:, 0],
                        kmf.confidence_interval_.iloc[:, 1], color=cols[g], alpha=0.12)
    g1 = d[d.grp == f"{gene}-high"]; g0 = d[d.grp == f"{gene}-low"]
    if len(g1) > 2 and len(g0) > 2:
        lr = logrank_test(g1[tcol], g0[tcol], event_observed_A=g1[ecol].astype(bool),
                          event_observed_B=g0[ecol].astype(bool))
        p = lr.p_value
    else:
        p = np.nan
    ax.set_xlabel("Time (years)", fontsize=8.5)
    ax.set_ylabel("OS probability", fontsize=8.5)
    ax.set_title(f"{title}\nlog-rank p = {p:.3g}", fontsize=9, fontweight="semibold")
    ax.grid(color=GRID, lw=0.6)
    ax.legend(fontsize=7.5, loc="upper right", frameon=False)
    return p

rows = []
for gene in GENES:
    for ep, tcol, ecol in [("OS", "OS_MONTHS", "OS_event"),
                           ("PFS", "PFS_MONTHS", "PFS_event"),
                           ("DSS", "DSS_MONTHS", "DSS_event")]:
        whole = cox_hr(df, gene, tcol, ecol)
        noRT = cox_hr(df[df.RTb == 0], gene, tcol, ecol)
        yesRT = cox_hr(df[df.RTb == 1], gene, tcol, ecol)
        it = interact(df, gene, tcol, ecol)
        n_no = int((df.RTb == 0).sum()); n_yes = int((df.RTb == 1).sum())
        rows.append(dict(gene=gene, endpoint=ep,
                         n_whole=int(df[tcol].notna().sum()),
                         HR_whole=whole["HR"] if whole else np.nan,
                         HR_RTno=noRT["HR"] if noRT else np.nan,
                         HR_RTyes=yesRT["HR"] if yesRT else np.nan,
                         n_noRT=n_no, n_RTyes=n_yes,
                         interact_HR=it["HR"] if it else np.nan,
                         interact_p=it["p"] if it else np.nan))
res = pd.DataFrame(rows)
res.to_csv(f"{ROOT}/interaction.csv", index=False)
print(res.to_string())

# KM figure: each gene: all / RT- / RT+ (OS)
ng = len(GENES)
fig = plt.figure(figsize=(10.2, 2.9 * ng + 1.0))
gs = fig.add_gridspec(ng, 3, hspace=0.78, wspace=0.26)
for ri, gene in enumerate(GENES):
    km_panel(fig, gs[ri, 0], df, gene, "OS_MONTHS", "OS_event", f"{gene} (all, n={df.OS_MONTHS.notna().sum()})")
    km_panel(fig, gs[ri, 1], df[df.RTb == 0], gene, "OS_MONTHS", "OS_event", f"{gene} RT− (n={(df.RTb==0).sum()})")
    km_panel(fig, gs[ri, 2], df[df.RTb == 1], gene, "OS_MONTHS", "OS_event", f"{gene} RT+ (n={(df.RTb==1).sum()})")
fig.suptitle("Gene expression and overall survival by radiotherapy status (TCGA-HNSC, median split)",
             fontsize=11.5, fontweight="bold", y=0.985)
fig.subplots_adjust(top=0.925, bottom=0.07)
fig.savefig(f"{ROOT}/OS_by_RT_KM.png", dpi=300, bbox_inches="tight")
print("saved OS_by_RT_KM.png")

# correlation with deconvolved fractions + HPV
frac_cols = ["prop_NK", "prop_Myeloid", "prop_Immune", "prop_Malignant", "Hypoxia_score"]
corr_rows = []
for gene in GENES:
    for c in frac_cols:
        sub = df[[gene, c, "HPV"]].dropna()
        if len(sub) < 5:
            continue
        r, p = stats.spearmanr(sub[gene], sub[c])
        corr_rows.append(dict(gene=gene, fraction=c, rho=r, p=p, n=len(sub)))
        for h in ["HPV+", "HPV-"]:
            s2 = sub[sub.HPV == h]
            if len(s2) > 5:
                r2, p2 = stats.spearmanr(s2[gene], s2[c])
                corr_rows.append(dict(gene=gene, fraction=f"{c}|{h}", rho=r2, p=p2, n=len(s2)))
corr = pd.DataFrame(corr_rows)
corr.to_csv(f"{ROOT}/fraction_corr.csv", index=False)
print(corr.to_string())

print("\nExpression by HPV:")
print(df.groupby("HPV")[GENES].agg(["mean", "median", "count"]).round(3).to_string())

# scatter gene vs NK fraction, colored by HPV
fig, axes = plt.subplots(1, len(GENES), figsize=(4.6 * len(GENES), 4.2))
fig.subplots_adjust(left=0.09, right=0.97, top=0.86, bottom=0.13, wspace=0.28)
if len(GENES) == 1:
    axes = [axes]
for ax, gene in zip(axes, GENES):
    s = df[[gene, "prop_NK", "HPV"]].dropna()
    for h, col in [("HPV+", C_HPVP), ("HPV-", C_HPVN)]:
        s2 = s[s.HPV == h]
        if len(s2):
            ax.scatter(s2[gene], s2.prop_NK * 100, s=18, color=col, alpha=0.55,
                       edgecolor="none", label=h)
    if len(s) > 5:
        rho, pp = stats.spearmanr(s[gene], s.prop_NK)
        ax.set_title(f"{gene} vs NK  (rho={rho:.2f}, p={pp:.1e})", fontsize=9.5, fontweight="semibold")
    ax.set_xlabel(f"{gene} expression (log2 RSEM+1)", fontsize=9)
    ax.set_ylabel("NK fraction (%)", fontsize=9)
    ax.grid(color=GRID, lw=0.6)
    ax.legend(fontsize=8)
fig.suptitle("Gene expression versus deconvolved NK fraction (TCGA-HNSC)",
             fontsize=11.5, fontweight="bold", y=0.97)
fig.savefig(f"{ROOT}/vs_NK_scatter.png", dpi=300, bbox_inches="tight")
print("saved vs_NK_scatter.png")
print("\nDONE ->", ROOT)
