#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
(1) NNLS deconvolution of TCGA-HNSC bulk using the GSE164690 cell-type
    reference, then split bulk DEPTOR into tumour vs myeloid/immune contributions.
(2) Survival / radiotherapy interaction of each contribution.
(3) Mediation: DEPTOR -> hypoxia (HIF1A program) -> radiotherapy benefit.

Deconvolution notes
-------------------
Reference H (7 types x 58 genes, mean log1p CP10K) and bulk Y (RSEM) live on
different scales, so each gene is divided by its reference mean before NNLS
(columns of H then average 1). Per sample we solve
    min_w ||H^T w - y||_2 ,  w >= 0
and normalise w to proportions.

DEPTOR split: contribution_c = w_c * DEPTOR_centroid_c ; the fraction of the
bulk DEPTOR signal attributable to tumour vs immune is that divided by the sum.
"""
import os, numpy as np, pandas as pd
from scipy.optimize import nnls
from scipy.stats import spearmanr, norm as normdist
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
import warnings; warnings.filterwarnings("ignore")

BASE = "WorkBuddy/2026-10-17-11-15-46"
OUT = os.path.join(BASE, "data", "results_deconv"); os.makedirs(OUT, exist_ok=True)

SETS = {
 "Tumor":  ["KRT5","KRT6A","KRT14","KRT8","KRT18","EPCAM","TP63","KRT19","SFN","KRT17"],
 "Myeloid":["LYZ","CD68","CD163","CSF1R","CD14","ITGAM","MRC1","VSIG4","AIF1","C1QA","C1QB",
            "FCGR1A","SPI1","LST1","TYROBP"],
 "Tcell":  ["CD3D","CD3E","CD8A","CD4","IL7R","CD2","CD3G","CD28"],
 "Bcell":  ["CD79A","MS4A1","JCHAIN","MZB1","CD19"],
 "NK":     ["NCAM1","NKG7","KLRF1","KLRD1","GNLY","KLRK1","PRF1"],
 "Stroma": ["COL1A1","COL1A2","ACTA2","PDGFRA","DCN","LUM","FAP","PDGFRB"],
 "Endo":   ["PECAM1","VWF","CLDN5","CDH5"],
 "Hypoxia":["HIF1A","CA9","VEGFA","SLC2A1","LDHA","PGK1","ADM","BNIP3","PDK1"],
}
IMMUNE_TYPES = ["Myeloid","Tcell","Bcell","NK"]

# ---------- load ----------
ref = pd.read_csv(os.path.join(BASE, "data", "gse164690_reference.csv"), index_col=0)
clin = pd.read_csv(os.path.join(BASE, "data", "tcga_hnsc_pca_deptor.csv"))
dec = pd.read_csv(os.path.join(BASE, "data", "tcga_hnsc_deconv_genes.csv"))
df = clin.merge(dec, on="sampleId", how="inner")
print("samples:", len(df), "| reference types:", list(ref.index))

genes = [g for g in ref.columns if g in df.columns and g != "DEPTOR"]
print(f"shared genes for deconvolution: {len(genes)}")
Hraw = ref[genes].values.astype(float)                          # 7 x G (log1p CP10K)
H = Hraw / np.maximum(Hraw.mean(axis=0, keepdims=True), 1e-9)   # gene mean -> 1
Y = df[genes].apply(pd.to_numeric, errors="coerce").values.astype(float)
Y = np.log2(np.nan_to_num(Y) + 1)                               # bulk RSEM is LINEAR -> log2
Yscaled = Y / np.maximum(Y.mean(axis=0, keepdims=True), 1e-9)   # gene mean -> 1 (own scale)

# ---------- NNLS per sample ----------
props = np.zeros((len(df), H.shape[0]))
for i in range(len(df)):
    w, _ = nnls(H.T, Yscaled[i])
    s = w.sum()
    props[i] = w / s if s > 0 else w
propdf = pd.DataFrame(props, columns=ref.index, index=df.index)
for c in propdf.columns:
    df[f"prop_{c}"] = propdf[c]
df["prop_Immune"] = propdf[IMMUNE_TYPES].sum(axis=1)
print("\nmean proportions:")
print(propdf.mean().round(3).to_string())

# ---------- split DEPTOR ----------
dC = ref["DEPTOR"].astype(float)
contrib = pd.DataFrame({c: propdf[c] * dC[c] for c in ref.index})
tot = contrib.sum(axis=1).replace(0, np.nan)
df["DEPTOR_tumor_frac"]  = contrib["Malignant"] / tot
df["DEPTOR_immune_frac"] = contrib[IMMUNE_TYPES].sum(axis=1) / tot
df["DEPTOR_myeloid_frac"]= contrib["Myeloid"] / tot
print("\nDEPTOR contribution fractions (mean):")
print(df[["DEPTOR_tumor_frac","DEPTOR_immune_frac","DEPTOR_myeloid_frac"]].mean().round(3).to_string())
print("\n-> tumour contributes {:.1%} of bulk DEPTOR signal on average".format(df.DEPTOR_tumor_frac.mean()))

# ---------- survival endpoints ----------
def ev(col):
    return df[col].astype(str).str.upper().str.strip().str.startswith("1").astype(int)
for t in ["OS","PFS","DSS"]:
    df[f"{t}_event"] = ev(f"{t}_STATUS")
    df[f"{t}_MONTHS"] = pd.to_numeric(df[f"{t}_MONTHS"], errors="coerce")
df["RT"] = df.RADIATION_THERAPY.astype(str).str.strip()
df["HPV"] = np.where(df.SUBTYPE.astype(str).str.contains("HPV-"), "HPV-",
            np.where(df.SUBTYPE.astype(str).str.contains("HPV\\+"), "HPV+", "NA"))
df["DEPTOR"] = pd.to_numeric(df.DEPTOR_RSEM, errors="coerce")

# hypoxia score
hyp = [g for g in SETS["Hypoxia"] if g in df.columns]
for g in hyp:
    v = pd.to_numeric(df[g], errors="coerce"); df[g] = v
    df[g+"_z"] = (v - v.mean())/v.std()
df["Hypoxia_score"] = df[[g+"_z" for g in hyp]].mean(axis=1)

# ---------- survival: tumour vs immune DEPTOR fraction ----------
print("\n=== Cox (RT=Yes): DEPTOR contribution fractions ===")
rt = df[df.RT == "Yes"].copy()
res = []
for var in ["DEPTOR_tumor_frac","DEPTOR_immune_frac","DEPTOR_myeloid_frac","DEPTOR"]:
    for t in ["OS","PFS","DSS"]:
        d = rt.dropna(subset=[t+"_MONTHS", t+"_event", var]).copy()
        if len(d) < 50: continue
        d["X"] = (d[var] - d[var].mean()) / d[var].std()      # per-SD, comparable across variables
        cph = CoxPHFitter()
        try:
            cph.fit(d[[t+"_MONTHS", t+"_event", "X"]], duration_col=t+"_MONTHS", event_col=t+"_event")
            s = cph.summary.loc["X"]
            res.append(dict(variable=var, endpoint=t, HR_perSD=round(s["exp(coef)"],3),
                            lo=round(s["exp(coef) lower 95%"],3), hi=round(s["exp(coef) upper 95%"],3),
                            p=round(s["p"],4), n=len(d)))
            print(f"  {var:22s} {t:4s} HR/SD={s['exp(coef)']:.2f} ({s['exp(coef) lower 95%']:.2f}-{s['exp(coef) upper 95%']:.2f}) p={s['p']:.4g} n={len(d)}")
        except Exception as e:
            print("  fail", var, t, str(e)[:60])
pd.DataFrame(res).to_csv(os.path.join(OUT, "deconv_survival.csv"), index=False)

# ---------- RT interaction ----------
print("\n=== DEPTOR_tumor_frac x RADIATION interaction (whole cohort) ===")
for t in ["OS","PFS","DSS"]:
    d = df.dropna(subset=[t+"_MONTHS", t+"_event", "DEPTOR_tumor_frac", "RT"]).copy()
    d["RT_yes"] = (d.RT == "Yes").astype(int)
    d["frac_z"] = (d.DEPTOR_tumor_frac - d.DEPTOR_tumor_frac.mean())/d.DEPTOR_tumor_frac.std()
    d["inter"] = d.frac_z * d.RT_yes
    cph = CoxPHFitter()
    try:
        cph.fit(d[[t+"_MONTHS", t+"_event", "frac_z","RT_yes","inter"]], duration_col=t+"_MONTHS", event_col=t+"_event")
        s = cph.summary
        it = s.loc["inter"]
        print(f"  {t:4s} interaction HR={it['exp(coef)']:.3f} ({it['exp(coef) lower 95%']:.3f}-{it['exp(coef) upper 95%']:.3f}) p={it['p']:.4g} n={len(d)}")
    except Exception as e:
        print("  fail", t, str(e)[:60])

# ---------- mediation: DEPTOR -> hypoxia -> survival ----------
print("\n=== Mediation: DEPTOR -> Hypoxia -> survival (RT=Yes) ===")
def fit_cox(d, cols):
    cph = CoxPHFitter()
    cph.fit(d[cols], duration_col=cols[0], event_col=cols[1])
    return cph

med_rows = []
for X in ["DEPTOR","DEPTOR_tumor_frac"]:
    for t in ["DSS","OS","PFS"]:
        d = rt.dropna(subset=[t+"_MONTHS", t+"_event", X, "Hypoxia_score"]).copy()
        if len(d) < 80: continue
        d["X"] = (d[X]-d[X].mean())/d[X].std()
        d["M"] = (d.Hypoxia_score-d.Hypoxia_score.mean())/d.Hypoxia_score.std()
        # a path: M ~ X
        import numpy as np
        a = np.polyfit(d["X"], d["M"], 1)[0]
        # c: total effect
        try:
            c_full = fit_cox(d[[t+"_MONTHS",t+"_event","X"]], [t+"_MONTHS",t+"_event","X"])
            lc = np.log(c_full.summary.loc["X","exp(coef)"]); pc = c_full.summary.loc["X","p"]
            # b and c'
            c2 = fit_cox(d[[t+"_MONTHS",t+"_event","X","M"]], [t+"_MONTHS",t+"_event","X","M"])
            lb = np.log(c2.summary.loc["M","exp(coef)"]); pb = c2.summary.loc["M","p"]
            lc2 = np.log(c2.summary.loc["X","exp(coef)"]); pc2 = c2.summary.loc["X","p"]
            med_prop = (lc - lc2)/lc if lc != 0 else np.nan
            med_rows.append(dict(X=X, endpoint=t, n=len(d), a_path=round(a,3),
                                 logHR_total=round(lc,3), p_total=round(pc,4),
                                 logHR_direct=round(lc2,3), p_direct=round(pc2,4),
                                 b_path_logHR=round(lb,3), p_b=round(pb,4),
                                 mediated_prop=round(med_prop,3)))
            print(f"  X={X:20s} {t:4s} n={len(d)}  a={a:+.2f}  total logHR={lc:+.3f}(p={pc:.3g})  "
                  f"direct={lc2:+.3f}(p={pc2:.3g})  b(logHR of M)={lb:+.3f}(p={pb:.3g})  mediated={med_prop:.1%}")
        except Exception as e:
            print("  med fail", X, t, str(e)[:70])
pd.DataFrame(med_rows).to_csv(os.path.join(OUT, "mediation.csv"), index=False)

# ---------- figures ----------
# (a) proportion stacked bar (by RT)
fig, ax = plt.subplots(figsize=(7,4))
order = ["Malignant","Stroma","Endo","Myeloid","Tcell","Bcell","NK"]
means = {rt_name: [df.loc[df.RT==rt_name, f"prop_{c}"].mean() for c in order]
         for rt_name in ["Yes","No"]}
x = np.arange(len(order)); w = 0.38
ax.bar(x-w/2, means["Yes"], w, label="RT=Yes", color="#d62728")
ax.bar(x+w/2, means["No"], w, label="RT=No", color="#1f77b4")
ax.set_xticks(x); ax.set_xticklabels(order, rotation=45, ha="right")
ax.set_ylabel("deconvolved proportion"); ax.set_title("TCGA-HNSC deconvolution (GSE164690 reference)")
ax.legend(fontsize=8); plt.tight_layout()
plt.savefig(os.path.join(OUT, "deconv_proportions.png"), dpi=140); plt.close()

# (b) DEPTOR contribution fractions
fig, ax = plt.subplots(figsize=(5.5,4))
ax.boxplot([df.DEPTOR_tumor_frac.dropna(), df.DEPTOR_immune_frac.dropna(), df.DEPTOR_myeloid_frac.dropna()],
           tick_labels=["tumour","immune","myeloid"])
ax.set_ylabel("fraction of bulk DEPTOR signal")
ax.set_title("Bulk DEPTOR is dominated by the tumour compartment")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "deconv_DEPTOR_fractions.png"), dpi=140); plt.close()

# (c) KM: tumour vs immune fraction high/low in RT=Yes
for var, tag in [("DEPTOR_tumor_frac","tumour"), ("DEPTOR_immune_frac","immune")]:
    d = rt.dropna(subset=["DSS_MONTHS","DSS_event", var]).copy()
    if len(d) < 40: continue
    d["g"] = np.where(d[var] > d[var].median(), "high","low")
    fig, ax = plt.subplots(figsize=(5.5,4.5))
    kmf = KaplanMeierFitter()
    for g in ["high","low"]:
        m = d.g == g
        kmf.fit(d.loc[m,"DSS_MONTHS"], d.loc[m,"DSS_event"], label=f"{g} (n={m.sum()})")
        kmf.plot_survival_function(ax=ax, ci_show=False)
    a = d[d.g=="high"]; b = d[d.g=="low"]
    lr = logrank_test(a["DSS_MONTHS"], b["DSS_MONTHS"], a["DSS_event"], b["DSS_event"])
    ax.set_title(f"RT=Yes, DSS by {tag} DEPTOR fraction\nlog-rank p={lr.p_value:.3g}", fontsize=9)
    ax.set_xlabel("DSS (months)"); plt.tight_layout()
    plt.savefig(os.path.join(OUT, f"km_RTyes_{tag}_DEPTORfrac_DSS.png"), dpi=140); plt.close()

# (d) DEPTOR vs hypoxia scatter (RT=Yes)
d = rt.dropna(subset=["DEPTOR","Hypoxia_score"])
fig, ax = plt.subplots(figsize=(5.5,4.5))
ax.scatter(d.DEPTOR, d.Hypoxia_score, s=14, alpha=.6,
           c=np.where(d.HPV=="HPV-","#d62728","#1f77b4"))
rho, p = spearmanr(d.DEPTOR, d.Hypoxia_score)
ax.set_xlabel("DEPTOR (log2 RSEM)"); ax.set_ylabel("Hypoxia score (z)")
ax.set_title(f"RT=Yes: DEPTOR vs hypoxia\nrho={rho:+.2f} (p={p:.2g}); red=HPV−, blue=HPV+", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "deconv_DEPTOR_vs_hypoxia.png"), dpi=140); plt.close()

df.to_csv(os.path.join(OUT, "deconv_per_sample.csv"), index=False)
print("\nSaved to", OUT, "DONE.")
