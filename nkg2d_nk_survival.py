#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test the hypothesis:

    myeloid DEPTOR high -> few NK
    radiotherapy up-regulates NKG2D ligands (MICA/B, ULBP1-3)
    -> but with few NK, the ligand signal has "no receiver"
    -> poorer radiotherapy benefit

TCGA-HNSC (PanCancer Atlas 2018).

Scores (mean z-score across genes):
  NKG2D_ligand : MICA, MICB, ULBP1, ULBP2, ULBP3, RAET1E(=ULBP4), RAET1G(=ULBP5),
                 RAET1L(=ULBP6)
  NK_signature : NKG7, GNLY, KLRD1, PRF1, FCGR3A, NCAM1, KLRF1, XCL1, XCL2, TBX21, EOMES

Analyses (RT-treated focus):
  1. NK low vs high  -> survival (expect NK-low worse)
  2. NKG2D-ligand high vs low -> survival
  3. 4 quadrants ligand x NK  -> KM; the "ligand-high / NK-low" quadrant is the
     "signal without receiver" group and should do worst.
  4. Cox interaction ligand_high:NK_low
  5. Relationship to DEPTOR; triple-high risk group.
  6. HPV stratification.
"""
import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
import warnings; warnings.filterwarnings("ignore")

BASE = "WorkBuddy/2026-10-17-11-15-46"
OUT = os.path.join(BASE, "data", "results_nkg2d"); os.makedirs(OUT, exist_ok=True)

LIG = ["MICA","MICB","ULBP1","ULBP2","ULBP3","RAET1E","RAET1G","RAET1L"]
NKG = ["NKG7","GNLY","KLRD1","PRF1","FCGR3A","NCAM1","KLRF1","XCL1","XCL2","TBX21","EOMES"]

clin = pd.read_csv(os.path.join(BASE, "data", "tcga_hnsc_pca_deptor.csv"))
nk = pd.read_csv(os.path.join(BASE, "data", "tcga_hnsc_nkg2d_nk.csv"))
df = clin.merge(nk, on="sampleId", how="inner")
print("merged samples:", len(df))

# ---- survival endpoints ----
# cBioPortal encodes these as "0:LIVING", "1:DECEASED", "1:DEAD WITH TUMOR",
# "1:PROGRESSION", "0:CENSORED" -> the leading 0/1 is the event flag.
def ev(col):
    s = df[col].astype(str).str.upper().str.strip()
    return s.str.startswith("1").astype(int)
df["OS_event"] = ev("OS_STATUS")
df["PFS_event"] = ev("PFS_STATUS")
df["DSS_event"] = ev("DSS_STATUS")
for t in ["OS","PFS","DSS"]:
    df[f"{t}_MONTHS"] = pd.to_numeric(df[f"{t}_MONTHS"], errors="coerce")
df["RT"] = df["RADIATION_THERAPY"].astype(str).str.strip()
df["HPV"] = np.where(df.SUBTYPE.astype(str).str.contains("HPV-"), "HPV-",
            np.where(df.SUBTYPE.astype(str).str.contains("HPV\\+"), "HPV+", "NA"))
df["DEPTOR"] = pd.to_numeric(df.DEPTOR_RSEM, errors="coerce")

# ---- scores ----
def zscore(col):
    v = pd.to_numeric(df[col], errors="coerce")
    df[col] = v
    return (v - v.mean()) / v.std()

for g in LIG + NKG:
    if g in df.columns:
        df[g + "_z"] = zscore(g)
lig_cols = [g + "_z" for g in LIG if g + "_z" in df.columns]
nk_cols  = [g + "_z" for g in NKG if g + "_z" in df.columns]
print(f"ligand genes used: {len(lig_cols)} | NK genes used: {len(nk_cols)}")
df["NKG2D_ligand"] = df[lig_cols].mean(axis=1)
df["NK_signature"] = df[nk_cols].mean(axis=1)

# ---- 1) correlations with DEPTOR ----
print("\n=== Spearman with DEPTOR ===")
for x in ["NKG2D_ligand","NK_signature"]:
    for sname, sub in [("All", df), ("RT=Yes", df[df.RT=="Yes"]), ("RT=No", df[df.RT=="No"])]:
        s = sub[["DEPTOR", x]].dropna()
        if len(s) > 10:
            rho, p = spearmanr(s.iloc[:,0], s.iloc[:,1])
            print(f"  DEPTOR vs {x:14s} [{sname:6s}] rho={rho:+.3f} (p={p:.3g}) n={len(s)}")

# ---- helper: KM + logrank by binary group ----
def km_by(group_col, t, sub, fname, title):
    d = sub.dropna(subset=[t+"_MONTHS", group_col]).copy()
    if len(d) < 20: return None
    fig, ax = plt.subplots(figsize=(5.5,4.5))
    kmf = KaplanMeierFitter()
    groups = sorted(d[group_col].dropna().unique())
    for g in groups:
        m = d[group_col] == g
        kmf.fit(d.loc[m, t+"_MONTHS"], d.loc[m, t+"_event"], label=f"{g} (n={m.sum()})")
        kmf.plot_survival_function(ax=ax, ci_show=False)
    if len(groups) == 2:
        a = d[d[group_col]==groups[0]]; b = d[d[group_col]==groups[1]]
        lr = logrank_test(a[t+"_MONTHS"], b[t+"_MONTHS"], a[t+"_event"], b[t+"_event"])
        ax.set_title(f"{title}\nlog-rank p={lr.p_value:.3g}", fontsize=9)
    else:
        ax.set_title(title, fontsize=9)
    ax.set_xlabel(t+" (months)"); ax.set_ylabel("survival")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, fname), dpi=140); plt.close()
    return d

rt = df[df.RT == "Yes"].copy()
# binary splits
df["LIG_hi"] = (df.NKG2D_ligand > df.NKG2D_ligand.median()).map({True:"ligand-high", False:"ligand-low"})
df["NK_hi"]  = (df.NK_signature > df.NK_signature.median()).map({True:"NK-high", False:"NK-low"})
rt = df[df.RT == "Yes"].copy()

print("\n=== KM in RT=Yes ===")
km_by("NK_hi", "DSS", rt, "km_RTyes_NK_DSS.png", "RT=Yes: NK signature (DSS)")
km_by("NK_hi", "PFS", rt, "km_RTyes_NK_PFS.png", "RT=Yes: NK signature (PFS)")
km_by("LIG_hi", "DSS", rt, "km_RTyes_ligand_DSS.png", "RT=Yes: NKG2D ligand (DSS)")

# ---- 4 quadrants ----
def quad(row):
    l = row["NKG2D_ligand"] > df["NKG2D_ligand"].median()
    n = row["NK_signature"] > df["NK_signature"].median()
    if l and n:   return "ligand-high/NK-high"
    if l and not n: return "ligand-high/NK-LOW"
    if not l and n: return "ligand-low/NK-high"
    return "ligand-low/NK-low"
df["quad"] = df.apply(quad, axis=1)
rt = df[df.RT == "Yes"].copy()

print("\n=== 4-quadrant group sizes (RT=Yes) ===")
print(rt["quad"].value_counts().to_string())

# KM by quadrant
d = rt.dropna(subset=["DSS_MONTHS"]).copy()
if len(d) >= 20:
    fig, ax = plt.subplots(figsize=(6.5,5))
    kmf = KaplanMeierFitter()
    for g in sorted(d["quad"].unique()):
        m = d["quad"] == g
        kmf.fit(d.loc[m,"DSS_MONTHS"], d.loc[m,"DSS_event"], label=f"{g} (n={m.sum()})")
        kmf.plot_survival_function(ax=ax, ci_show=False)
    # logrank: key quadrant vs rest
    key = "ligand-high/NK-LOW"
    a = d[d.quad == key]; b = d[d.quad != key]
    lr = logrank_test(a["DSS_MONTHS"], b["DSS_MONTHS"], a["DSS_event"], b["DSS_event"])
    ax.set_title(f"RT=Yes, DSS by ligand x NK quadrant\n'{key}' vs rest: log-rank p={lr.p_value:.3g}", fontsize=9)
    ax.set_xlabel("DSS (months)"); ax.set_ylabel("survival")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "km_RTyes_quadrant_DSS.png"), dpi=140); plt.close()

# ---- Cox interaction ----
print("\n=== Cox: ligand_high * NK_low interaction (RT=Yes) ===")
cox_rows = []
for t in ["OS","PFS","DSS"]:
    d = rt.dropna(subset=[t+"_MONTHS", t+"_event", "NKG2D_ligand", "NK_signature"]).copy()
    if len(d) < 40: continue
    d["ligand_hi"] = (d.NKG2D_ligand > d.NKG2D_ligand.median()).astype(int)
    d["NK_low"] = (d.NK_signature <= d.NK_signature.median()).astype(int)
    # main effects model with interaction
    d2 = d[[t+"_MONTHS", t+"_event", "ligand_hi", "NK_low"]].copy()
    d2["inter"] = d2.ligand_hi * d2.NK_low
    cph = CoxPHFitter()
    try:
        cph.fit(d2, duration_col=t+"_MONTHS", event_col=t+"_event")
        s = cph.summary
        for term in ["ligand_hi","NK_low","inter"]:
            if term in s.index:
                cox_rows.append(dict(endpoint=t, term=term,
                    HR=round(s.loc[term,"exp(coef)"],3),
                    lo=round(s.loc[term,"exp(coef) lower 95%"],3),
                    hi=round(s.loc[term,"exp(coef) upper 95%"],3),
                    p=round(s.loc[term,"p"],4), n=len(d2)))
    except Exception as e:
        print("  cox fail", t, e)
cox = pd.DataFrame(cox_rows)
if len(cox):
    cox.to_csv(os.path.join(OUT, "cox_interaction.csv"), index=False)
    print(cox.to_string(index=False))

# ---- NK-low vs NK-high HR (univariate) ----
print("\n=== Univariate Cox: NK-low and ligand-high in RT=Yes ===")
uni = []
for t in ["OS","PFS","DSS"]:
    d = rt.dropna(subset=[t+"_MONTHS", t+"_event", "NKG2D_ligand", "NK_signature"]).copy()
    if len(d) < 40: continue
    d["NK_low"] = (d.NK_signature <= d.NK_signature.median()).astype(int)
    d["ligand_hi"] = (d.NKG2D_ligand > d.NKG2D_ligand.median()).astype(int)
    for var in ["NK_low","ligand_hi"]:
        cph = CoxPHFitter()
        try:
            cph.fit(d[[t+"_MONTHS", t+"_event", var]], duration_col=t+"_MONTHS", event_col=t+"_event")
            s = cph.summary.loc[var]
            uni.append(dict(endpoint=t, var=var, HR=round(s["exp(coef)"],3),
                            lo=round(s["exp(coef) lower 95%"],3), hi=round(s["exp(coef) upper 95%"],3),
                            p=round(s["p"],4), n=len(d)))
            print(f"  {t:4s} {var:10s} HR={s['exp(coef)']:.2f} ({s['exp(coef) lower 95%']:.2f}-{s['exp(coef) upper 95%']:.2f}) p={s['p']:.4g} n={len(d)}")
        except Exception as e:
            print("  fail", t, var, e)
pd.DataFrame(uni).to_csv(os.path.join(OUT, "univariate_cox.csv"), index=False)

# ---- HPV stratification ----
print("\n=== HPV stratification (RT=Yes): NK-low HR (DSS) ===")
for h in ["HPV-","HPV+"]:
    d = rt[rt.HPV == h].dropna(subset=["DSS_MONTHS","DSS_event","NK_signature"]).copy()
    if len(d) < 30: 
        print(f"  {h}: n={len(d)} too few"); continue
    d["NK_low"] = (d.NK_signature <= d.NK_signature.median()).astype(int)
    cph = CoxPHFitter()
    try:
        cph.fit(d[["DSS_MONTHS","DSS_event","NK_low"]], duration_col="DSS_MONTHS", event_col="DSS_event")
        s = cph.summary.loc["NK_low"]
        print(f"  {h}: NK-low HR={s['exp(coef)']:.2f} ({s['exp(coef) lower 95%']:.2f}-{s['exp(coef) upper 95%']:.2f}) p={s['p']:.4g} n={len(d)}")
    except Exception as e:
        print("  fail", h, e)

# ---- figures: ligand vs NK scatter (colored by RT), DEPTOR vs NK ----
fig, axes = plt.subplots(1, 2, figsize=(11,4.5))
ax = axes[0]
for lab, m, c in [("RT=Yes", df.RT=="Yes", "#d62728"), ("RT=No", df.RT=="No", "#1f77b4")]:
    ax.scatter(df.loc[m,"NK_signature"], df.loc[m,"NKG2D_ligand"], s=12, alpha=.5, c=c, label=lab)
ax.axhline(df.NKG2D_ligand.median(), color="k", lw=.6, ls="--")
ax.axvline(df.NK_signature.median(), color="k", lw=.6, ls="--")
ax.set_xlabel("NK signature (z)"); ax.set_ylabel("NKG2D ligand (z)")
ax.set_title("NKG2D ligand vs NK infiltration"); ax.legend(fontsize=8)

ax = axes[1]
ax.scatter(df["DEPTOR"], df["NK_signature"], s=12, alpha=.5, c=np.where(df.RT=="Yes","#d62728","#1f77b4"))
rho, p = spearmanr(df[["DEPTOR","NK_signature"]].dropna().iloc[:,0], df[["DEPTOR","NK_signature"]].dropna().iloc[:,1])
ax.set_xlabel("DEPTOR (log2 RSEM)"); ax.set_ylabel("NK signature (z)")
ax.set_title(f"DEPTOR vs NK signature\nrho={rho:+.2f} (p={p:.2g})", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "nkg2d_ligand_NK_DEPTOR.png"), dpi=140); plt.close()

print("\nSaved to", OUT, "DONE.")
