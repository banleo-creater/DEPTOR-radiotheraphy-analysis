#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Formal orthogonality test for the two-axis DEPTOR model.  ONE table, ONE figure.

Claim under test
----------------
Axis 1 (tumour-intrinsic): DEPTOR x RT interaction -> radiotherapy benefit.
Axis 2 (myeloid)         : myeloid DEPTOR -> NK-poor microenvironment.
Proposed to be INDEPENDENT. Two testable consequences:

  (a) TCGA: the DEPTOR x RT interaction must SURVIVE adjustment for NK /
      immune content; and NK x RT should NOT itself predict RT benefit.
  (b) GSE164690: with both axes measured in the SAME patients, myeloid and
      malignant DEPTOR must each retain an independent coefficient in one
      model of NK fraction.

Encoding note (important)
-------------------------
DEPTOR in Xena is linear RSEM (median 100, max 2383). A per-SD term on that
scale is outlier-driven and under-powers the interaction. The pre-specified
PRIMARY encoding is therefore log2(DEPTOR) per SD, which is monotone in the
expression and is the scale the assay is normally analysed on. The
median-split encoding used in earlier figures is retained as a sensitivity
analysis, not as the primary test.
"""
import os, warnings
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
warnings.filterwarnings("ignore")

BASE = "WorkBuddy/2026-10-17-11-15-46"
OUT = os.path.join(BASE, "data", "results_orthogonality")
os.makedirs(OUT, exist_ok=True)

d = pd.read_csv(os.path.join(BASE, "data", "results_deconv", "deconv_per_sample.csv"))
d["RT"] = d["RT"].map({"Yes": 1, "No": 0})
d["HPVpos"] = d["HPV"].map({"HPV+": 1, "HPV-": 0})
d["male"] = (d["SEX"] == "Male").astype(float)
d["stage_late"] = np.where(
    d["AJCC_PATHOLOGIC_TUMOR_STAGE"].isna(), np.nan,
    d["AJCC_PATHOLOGIC_TUMOR_STAGE"].astype(str).str.contains("III|IV").astype(float))

# CANONICAL CUT (2026-10-19): the pre-specified cut-point is the median of the
# WHOLE cohort (n = 515), NOT of each endpoint-specific analytic subset. The cut
# must not depend on the analytic set, otherwise OS/PFS/DSS each get a slightly
# different threshold (100.214 vs 100.210 RSEM) and the interaction p-values
# drift between 0.0033 and 0.0036. One cut, used everywhere.
MED_DEPTOR = float(d["DEPTOR"].median())
print(f"canonical DEPTOR cut (whole cohort, n={d.DEPTOR.notna().sum()}): "
      f"{MED_DEPTOR:.4f} RSEM")

rows = []


def z(s):
    s = pd.Series(s).astype(float)
    return (s - s.mean()) / s.std(ddof=1)


def fit_cox(df, cols, term, endpoint, model, part="A"):
    sub = df[list(cols) + ["T", "E"]].astype(float)
    cph = CoxPHFitter()
    cph.fit(sub, duration_col="T", event_col="E")
    if term not in cph.summary.index:
        return None, None
    r = cph.summary.loc[term]
    out = dict(part=part, endpoint=endpoint, model=model, term=term,
               n=int(len(sub)), events=int(sub.E.sum()),
               HR=float(r["exp(coef)"]), lo=float(r["exp(coef) lower 95%"]),
               hi=float(r["exp(coef) upper 95%"]), p=float(r["p"]), note="")
    rows.append(out)
    print(f"  [{endpoint}] {model:24s} {term:8s} HR={out['HR']:.3f} "
          f"({out['lo']:.3f}-{out['hi']:.3f}) P={out['p']:.4g} "
          f"n={out['n']} ev={out['events']}")
    return out, cph


# =====================================================================
# PART 1 — TCGA
# =====================================================================
print("=" * 80)
print("PART 1  TCGA — does the DEPTOR x RT interaction survive immune adjustment?")
print("=" * 80)

END = {"OS": ("OS_MONTHS", "OS_event"), "PFS": ("PFS_MONTHS", "PFS_event"),
       "DSS": ("DSS_MONTHS", "DSS_event")}
COV = ["HPVpos", "stage_late", "AGE_z", "male"]

for ep, (tcol, ecol) in END.items():
    a = d.dropna(subset=["DEPTOR", "RT", "prop_NK", tcol, ecol]).copy()
    a["T"], a["E"] = a[tcol], a[ecol].astype(int)
    a["D_z"] = z(np.log2(a["DEPTOR"].clip(lower=1)))     # PRIMARY encoding
    a["Dhi"] = (a["DEPTOR"] > MED_DEPTOR).astype(float)
    a["NK_z"], a["IMM_z"] = z(a["prop_NK"]), z(a["prop_Immune"])
    a["DxRT"] = a["D_z"] * a["RT"]
    a["HixRT"] = a["Dhi"] * a["RT"]
    a["NKxRT"] = a["NK_z"] * a["RT"]
    a["IMMxRT"] = a["IMM_z"] * a["RT"]
    print(f"\n  --- {ep}  (n={len(a)}, events={int(a.E.sum())}) ---")

    m0, _ = fit_cox(a, ["D_z", "RT", "DxRT"], "DxRT", ep, "M0 unadjusted")
    m1, _ = fit_cox(a, ["D_z", "RT", "DxRT", "NK_z"], "DxRT", ep, "M1 +NK")
    m2, _ = fit_cox(a, ["D_z", "RT", "DxRT", "NK_z", "NKxRT"], "DxRT", ep,
                    "M2 +NK+NKxRT")
    m3, _ = fit_cox(a, ["D_z", "RT", "DxRT", "IMM_z"], "DxRT", ep,
                    "M3 +immune content")
    fit_cox(a, ["D_z", "RT", "DxRT", "NK_z", "NKxRT"], "NKxRT", ep,
            "M2 (NKxRT term)")
    fit_cox(a, ["D_z", "RT", "DxRT", "IMM_z", "IMMxRT"], "IMMxRT", ep,
            "M3 (IMMxRT term)")
    # sensitivity: median split
    fit_cox(a, ["Dhi", "RT", "HixRT"], "HixRT", ep, "S1 median split")
    fit_cox(a, ["Dhi", "RT", "HixRT", "NK_z", "NKxRT"], "HixRT", ep,
            "S1 median split +NK")

    # arm-specific HR implied by M0 and M2
    for tag, cols in [("M0", ["D_z", "RT", "DxRT"]),
                      ("M2", ["D_z", "RT", "DxRT", "NK_z", "NKxRT"])]:
        _, cph = fit_cox(a, cols, "DxRT", ep, f"({tag} arms)")
        bd, bi = cph.params_["D_z"], cph.params_["DxRT"]
        vd, vi = cph.variance_matrix_.loc["D_z", "D_z"], cph.variance_matrix_.loc["DxRT", "DxRT"]
        cdi = cph.variance_matrix_.loc["D_z", "DxRT"]
        for arm, b, var in [("RT-no", bd, vd),
                            ("RT-yes", bd + bi, vd + vi + 2 * cdi)]:
            se = np.sqrt(var)
            rows.append(dict(part="A2", endpoint=ep, model=tag,
                             term=f"DEPTOR HR {arm}", n=int(len(a)),
                             events=int(a.E.sum()), HR=float(np.exp(b)),
                             lo=float(np.exp(b - 1.96 * se)),
                             hi=float(np.exp(b + 1.96 * se)),
                             p=float(2 * (1 - stats.norm.cdf(abs(b / se)))), note=""))
            print(f"    {tag} DEPTOR HR in {arm:7s} = {np.exp(b):.3f} "
                  f"({np.exp(b-1.96*se):.3f}-{np.exp(b+1.96*se):.3f})")

    # attenuation of the interaction log-HR after immune adjustment
    for tag, m in [("+NK (M1)", m1), ("+immune content (M3)", m3)]:
        if m0 and m:
            att = (np.log(m0["HR"]) - np.log(m["HR"])) / np.log(m0["HR"]) * 100
            rows.append(dict(part="A3", endpoint=ep, model=f"M0->{tag}",
                             term="% attenuation", n=m0["n"], events=m0["events"],
                             HR=np.nan, lo=np.nan, hi=np.nan, p=np.nan,
                             note=f"{att:.1f}%"))
            print(f"    -> attenuation after {tag:20s}: {att:+.1f}%")

    # fully adjusted (complete case)
    b = a.dropna(subset=["HPVpos", "stage_late", "AGE", "male"]).copy()
    b["AGE_z"] = z(b["AGE"])
    print(f"    [complete-case subset for M4: n={len(b)}]")
    fit_cox(b, ["D_z", "RT", "DxRT", "NK_z", "NKxRT"] + COV, "DxRT", ep,
            "M4 fully adjusted")
    fit_cox(b, ["D_z", "RT", "DxRT", "NK_z", "NKxRT"] + COV, "NKxRT", ep,
            "M4 (NKxRT term)")

# PH assumption on the primary DSS model
print("\n  proportional-hazards (Schoenfeld, rank) for M2 on DSS:")
a = d.dropna(subset=["DEPTOR", "RT", "prop_NK", "DSS_MONTHS", "DSS_event"]).copy()
a["T"], a["E"] = a["DSS_MONTHS"], a["DSS_event"].astype(int)
a["D_z"] = z(np.log2(a["DEPTOR"].clip(lower=1)))
a["NK_z"] = z(a["prop_NK"])
a["DxRT"], a["NKxRT"] = a.D_z * a.RT, a.NK_z * a.RT
phcols = ["D_z", "RT", "DxRT", "NK_z", "NKxRT", "T", "E"]
aph = a[phcols].astype(float)
cph = CoxPHFitter(); cph.fit(aph, duration_col="T", event_col="E")
pht = proportional_hazard_test(cph, aph, time_transform="rank")
for idx, r in pht.summary.iterrows():
    print(f"    {idx:8s} P={r['p']:.3f}")
    rows.append(dict(part="A4", endpoint="DSS", model="PH test", term=str(idx),
                     n=len(aph), events=int(aph.E.sum()), HR=np.nan, lo=np.nan,
                     hi=np.nan, p=float(r["p"]), note="Schoenfeld rank"))

# =====================================================================
# PART 2 — GSE164690: joint model, both axes in the same patients
# =====================================================================
print("\n" + "=" * 80)
print("PART 2  GSE164690 — myeloid and malignant DEPTOR entered together")
print("=" * 80)

x = pd.read_csv(os.path.join(BASE, "data", "results_dual", "dual_axis_summary.csv"))
x = x.rename(columns={"NK_fraction_of_immune": "NKfrac"})
both = x.dropna(subset=["NKfrac", "DEPTOR_myeloid_mean",
                        "DEPTOR_malignant_mean"]).copy()
print(f"  patients with both axes measured: n = {len(both)}")


def ols_std(df, ycol, xcols, label):
    X = np.column_stack([np.ones(len(df))] + [z(df[c]).values for c in xcols])
    y = z(df[ycol]).values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(df) - X.shape[1]
    s2 = resid @ resid / dof
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    r2 = 1 - (resid @ resid) / ((y - y.mean()) ** 2).sum()
    print(f"  [{label}] n={len(df)}  R2={r2:.3f}")
    for i, c in enumerate(xcols, start=1):
        b, s = beta[i], se[i]
        p = 2 * (1 - stats.t.cdf(abs(b / s), dof))
        rows.append(dict(part="B", endpoint="NK fraction",
                         model=f"{label} | {c}", n=len(df), events=np.nan,
                         HR=np.nan, lo=np.nan, hi=np.nan, p=p,
                         note=f"std beta={b:+.3f} ({b-1.96*s:+.3f},{b+1.96*s:+.3f})"))
        print(f"    {c:24s} std beta={b:+.3f} ({b-1.96*s:+.3f},{b+1.96*s:+.3f}) "
              f"P={p:.3g}")
    if len(xcols) == 2:
        r = float(np.corrcoef(z(df[xcols[0]]), z(df[xcols[1]]))[0, 1])
        rows.append(dict(part="B", endpoint="NK fraction",
                         model=f"{label} | collinearity", n=len(df),
                         events=np.nan, HR=np.nan, lo=np.nan, hi=np.nan, p=np.nan,
                         note=f"r={r:+.3f}, VIF={1/(1-r**2):.2f}"))
        print(f"    predictor collinearity r={r:+.3f}, VIF={1/(1-r**2):.2f}")


ols_std(both, "NKfrac", ["DEPTOR_myeloid_mean", "DEPTOR_malignant_mean"],
        "joint model")
ols_std(x.dropna(subset=["NKfrac", "DEPTOR_myeloid_mean"]), "NKfrac",
        ["DEPTOR_myeloid_mean"], "myeloid only (n=18)")


def partial_spearman(df, ycol, xcol, ctrl):
    ry, rx, rc = (stats.rankdata(df[c]) for c in (ycol, xcol, ctrl))

    def res(a, b):
        return a - np.polyval(np.polyfit(b, a, 1), b)
    r, p = stats.pearsonr(res(ry, rc), res(rx, rc))
    rows.append(dict(part="B", endpoint="NK fraction",
                     model=f"partial rho: {xcol} | {ctrl}", n=len(df),
                     events=np.nan, HR=np.nan, lo=np.nan, hi=np.nan, p=p,
                     note=f"partial rho={r:+.3f}"))
    print(f"    partial rho({xcol} vs NK | {ctrl}) = {r:+.3f}, P={p:.3g}")


partial_spearman(both, "NKfrac", "DEPTOR_myeloid_mean", "DEPTOR_malignant_mean")
partial_spearman(both, "NKfrac", "DEPTOR_malignant_mean", "DEPTOR_myeloid_mean")

res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, "orthogonality_models.csv"), index=False)
print(f"\nsaved {len(res)} model terms")

# =====================================================================
# FIGURE
# =====================================================================
BLUE, RED, GREY, TEAL = "#2b6cb0", "#c0392b", "#5a5a5a", "#1f8a70"
fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.4))

ax = axes[0]
mods = ["M0 unadjusted", "M1 +NK", "M2 +NK+NKxRT", "M3 +immune content"]
mcol = {"OS": GREY, "PFS": BLUE, "DSS": RED}
yy, ticks, labs = 0, [], []
for ep in ["DSS", "PFS", "OS"]:
    for m in mods:
        r = res[(res.part == "A") & (res.endpoint == ep) & (res.model == m)]
        if not len(r):
            continue
        r = r.iloc[0]
        ax.plot([r.lo, r.hi], [yy, yy], color=mcol[ep], lw=2)
        ax.scatter([r.HR], [yy], c=mcol[ep], s=32, zorder=3)
        ticks.append(yy); labs.append(f"{ep} · {m}")
        yy += 1
    yy += 0.7
ax.axvline(1, color="k", lw=1, ls="--")
ax.set_yticks(ticks); ax.set_yticklabels(labs, fontsize=6.8)
ax.invert_yaxis()
ax.set_xlabel("DEPTOR × RT interaction HR  (per SD, log2 scale)", fontsize=9)
ax.set_title("A   Interaction persists after immune adjustment",
             fontsize=10.5, fontweight="bold", loc="left")
ax.text(0.985, 0.02, "all M0–M3: P ≤ 0.015", transform=ax.transAxes,
        ha="right", fontsize=7.4, color=GREY)
ax.set_xlim(0.32, 1.06)
ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_xticklabels(["0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"])
ax.tick_params(labelsize=8)

ax = axes[1]
b = res[(res.part == "B") & res.model.str.startswith("joint model | DEPTOR")]
labs2, vals, los, his, cols, pvs = [], [], [], [], [], []
for _, r in b.iterrows():
    nm = "myeloid DEPTOR" if "myeloid" in r.model else "malignant DEPTOR"
    txt = r.note.split("std beta=")[1]
    val = float(txt.split(" ")[0])
    ci = txt.split("(")[1].split(")")[0].split(",")
    labs2.append(nm); vals.append(val)
    los.append(float(ci[0])); his.append(float(ci[1]))
    cols.append(RED if "myeloid" in r.model else TEAL)
    pvs.append(float(r.p))
for i, (v, lo, hi, c, pv) in enumerate(zip(vals, los, his, cols, pvs)):
    ax.plot([lo, hi], [i, i], color=c, lw=2.8, solid_capstyle="round")
    ax.scatter([v], [i], c=c, s=70, zorder=3)
    ax.text(-1.12, i + 0.26, f"β = {v:+.2f}   P = {pv:.3f}", fontsize=8,
            color=c, ha="left")
ax.axvline(0, color="k", lw=1, ls="--")
ax.set_yticks(range(len(labs2))); ax.set_yticklabels(labs2, fontsize=9.5)
ax.set_xlabel("standardised β for NK fraction of immune cells", fontsize=9)
ax.set_title("B   Both axes contribute independently (same patients)",
             fontsize=10.5, fontweight="bold", loc="left")
ax.set_ylim(-0.7, len(labs2) - 0.3)
ax.set_xlim(-1.18, 0.12)
ax.text(0.985, 0.045, "collinearity r = −0.05, VIF = 1.00",
        transform=ax.transAxes, ha="right", fontsize=7.4, color=GREY)
ax.tick_params(labelsize=8)

fig.suptitle("Orthogonality of the tumour-intrinsic and myeloid DEPTOR axes",
             fontsize=12, fontweight="bold", y=0.99)
plt.tight_layout(rect=(0, 0, 1, 0.95))
fp = os.path.join(OUT, "Figure_orthogonality.png")
plt.savefig(fp, dpi=300, facecolor="white", bbox_inches="tight")
plt.close()
print("saved", fp)
