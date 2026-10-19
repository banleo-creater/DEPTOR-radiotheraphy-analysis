#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Consistent, three-cohort analysis of
  (i)  the DEPTOR x radiotherapy interaction on overall survival, and
  (ii) mediation of that interaction by the HIF1A / hypoxia programme.

Cohorts
  TCGA-HNSC   (n=515, RNA-seq, adjuvant RT yes/no, OS + DSS)
  GSE65858    (n=270, Illumina array, multi- vs uni-modality, OS + PFS)
  GSE41613    (n=97 , Affy U133Plus2, multi- vs uni-modality, OS, all HPV-)

Everything is refitted here from the per-sample tables so the three cohorts are
handled identically (per-SD DEPTOR, same Cox specification, same mediator).
"""
import os, numpy as np, pandas as pd
from scipy.stats import norm, spearmanr
from lifelines import CoxPHFitter
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore")

BASE = "WorkBuddy/2026-10-17-11-15-46"
OUT = os.path.join(BASE, "data", "results_pooled3"); os.makedirs(OUT, exist_ok=True)


def zs(s):
    s = pd.to_numeric(s, errors="coerce")
    return (s - s.mean()) / s.std()


def fit_cox(d, covs, t, e):
    d = d.dropna(subset=[t, e] + covs).copy()
    if len(d) < 30 or d[e].sum() < 8:
        return None
    cph = CoxPHFitter()
    cph.fit(d[[t, e] + covs], duration_col=t, event_col=e)
    return cph, d


def interaction(d, t, e, treat, extra=None):
    """logHR of DEPTOR_z x treatment, with and without the mediator handled outside"""
    d = d.copy()
    d["Dz"] = zs(d["DEPTOR"])
    d["inter"] = d["Dz"] * d[treat]
    covs = ["Dz", treat, "inter"] + (extra or [])
    r = fit_cox(d, covs, t, e)
    if r is None:
        return None
    cph, dd = r
    s = cph.summary.loc["inter"]
    return dict(logHR=s["coef"], se=s["se(coef)"], p=s["p"],
                HR=np.exp(s["coef"]), lo=np.exp(s["coef lower 95%"]),
                hi=np.exp(s["coef upper 95%"]), n=len(dd), events=int(dd[e].sum()),
                logHR_adj=s["coef"])


def arm_hr(d, t, e, treat, arm):
    d = d.copy(); d["Dz"] = zs(d["DEPTOR"])
    r = fit_cox(d[d[treat] == arm], ["Dz"], t, e)
    if r is None: return None
    cph, dd = r; s = cph.summary.loc["Dz"]
    return dict(HR=np.exp(s["coef"]), lo=np.exp(s["coef lower 95%"]),
                hi=np.exp(s["coef upper 95%"]), p=s["p"], n=len(dd), events=int(dd[e].sum()))


# ------------------------------------------------------------------ load
def load_tcga():
    a = pd.read_csv(os.path.join(BASE, "data", "results_deconv", "deconv_per_sample.csv"), index_col=0)
    d = pd.DataFrame(index=a.index)
    d["DEPTOR"] = a["DEPTOR"]
    d["hypoxia"] = a["Hypoxia_score"]
    d["NK_sig"] = a[[c for c in a.columns if c.endswith("_z") and c.split("_z")[0] in
                     ("HIF1A", "CA9", "VEGFA", "SLC2A1", "LDHA", "PGK1", "ADM", "BNIP3", "PDK1")]].mean(axis=1) \
        if "Hypoxia_score" not in a else a["Hypoxia_score"]
    d["os_t"] = pd.to_numeric(a["OS_MONTHS"], errors="coerce")
    d["os_e"] = a["OS_STATUS"].astype(str).str.startswith("1").astype(int)
    d["dss_t"] = pd.to_numeric(a["DSS_MONTHS"], errors="coerce")
    d["dss_e"] = a["DSS_STATUS"].astype(str).str.startswith("1").astype(int)
    d["treat"] = (a["RADIATION_THERAPY"].astype(str).str.lower() == "yes").astype(int)
    d.loc[a["RADIATION_THERAPY"].isna(), "treat"] = np.nan
    d["hpv_neg"] = a["SUBTYPE"].astype(str).str.contains("HPV-", na=False)
    d["stage3"] = a["AJCC_PATHOLOGIC_TUMOR_STAGE"].astype(str).str.contains("IV", na=False).astype(int)
    return d.dropna(subset=["DEPTOR", "os_t", "os_e"])


def load_65858():
    a = pd.read_csv(os.path.join(BASE, "data", "results_deconv65858", "gse65858_deconv_sample.csv"), index_col=0)
    d = pd.DataFrame(index=a.index)
    d["DEPTOR"] = a["DEPTOR"]; d["hypoxia"] = a["hypoxia"]; d["NK_sig"] = a["NK_sig"]
    d["os_t"] = pd.to_numeric(a["os"], errors="coerce")
    d["os_e"] = pd.to_numeric(a["os_event"], errors="coerce")
    d["treat"] = a["treatment"].astype(str).str.lower().str.startswith("multi").astype(int)
    d["dss_t"] = np.nan; d["dss_e"] = np.nan; d["hpv_neg"] = np.nan
    d["stage3"] = np.nan
    return d.dropna(subset=["DEPTOR", "os_t", "os_e"])


def load_41613():
    a = pd.read_csv(os.path.join(BASE, "data", "results_deconv41613", "GSE41613_deconv_sample.csv"), index_col=0)
    d = pd.DataFrame(index=a.index)
    d["DEPTOR"] = a["DEPTOR"]; d["hypoxia"] = a["hypoxia"]; d["NK_sig"] = a["NK_sig"]
    d["os_t"] = a["os_t"]; d["os_e"] = a["os_e"]
    d["treat"] = a["multi"]
    d["dss_t"] = np.nan; d["dss_e"] = np.nan
    d["hpv_neg"] = True; d["stage3"] = a["stage3"]
    return d.dropna(subset=["DEPTOR", "os_t", "os_e"])


COH = {"TCGA-HNSC": load_tcga(), "GSE65858": load_65858(), "GSE41613": load_41613()}

# ------------------------------------------------------------------ per cohort
rows, arms, med = [], [], []
for name, d in COH.items():
    for endpoint, (t, e) in {"OS": ("os_t", "os_e"), "DSS": ("dss_t", "dss_e")}.items():
        if d[t].notna().sum() < 40: continue
        r = interaction(d, t, e, "treat")
        if r is None: continue
        rows.append(dict(cohort=name, endpoint=endpoint, **{k: (round(v, 4) if isinstance(v, float) else v)
                                                            for k, v in r.items()}))
        print(f"[{name}] {endpoint}: interaction HR={r['HR']:.2f} ({r['lo']:.2f}-{r['hi']:.2f}) "
              f"p={r['p']:.3g} n={r['n']} ev={r['events']}")
    for endpoint, (t, e) in {"OS": ("os_t", "os_e")}.items():
        for arm, lbl in [(0, "no RT / uni"), (1, "RT / multi")]:
            r = arm_hr(d, t, e, "treat", arm)
            if r: arms.append(dict(cohort=name, endpoint=endpoint, arm=lbl, **{k: round(v, 4) for k, v in r.items()}))
    # ---- mediation on OS, with hypoxia as mediator
    dd = d.dropna(subset=["os_t", "os_e", "DEPTOR", "hypoxia", "treat"]).copy()
    if len(dd) >= 40:
        dd["Dz"] = zs(dd["DEPTOR"]); dd["Mz"] = zs(dd["hypoxia"])
        dd["inter"] = dd["Dz"] * dd["treat"]
        a_coef = np.polyfit(dd["Dz"], dd["Mz"], 1)[0]
        rho, prho = spearmanr(dd["Dz"], dd["Mz"])
        c1 = CoxPHFitter(); c1.fit(dd[["os_t", "os_e", "Dz", "treat", "inter"]], "os_t", "os_e")
        c2 = CoxPHFitter(); c2.fit(dd[["os_t", "os_e", "Dz", "treat", "inter", "Mz"]], "os_t", "os_e")
        th1 = c1.summary.loc["inter", "coef"]; th2 = c2.summary.loc["inter", "coef"]
        b = c2.summary.loc["Mz"]
        # mediate: DEPTOR -> hypoxia -> OS (main effect, within RT arm)
        rt = dd[dd.treat == 1]
        medprop = np.nan
        if len(rt) >= 35 and rt.os_e.sum() >= 8:
            m1 = CoxPHFitter(); m1.fit(rt[["os_t", "os_e", "Dz"]], "os_t", "os_e")
            m2 = CoxPHFitter(); m2.fit(rt[["os_t", "os_e", "Dz", "Mz"]], "os_t", "os_e")
            tot = m1.summary.loc["Dz", "coef"]; dr = m2.summary.loc["Dz", "coef"]
            medprop = (tot - dr) / tot if abs(tot) > 1e-6 else np.nan
        med.append(dict(cohort=name, n=len(dd), a_path=round(a_coef, 3), a_rho=round(rho, 3), a_p=round(prho, 5),
                        inter_logHR=round(th1, 4), inter_logHR_adj_M=round(th2, 4),
                        attenuation=round((th1 - th2) / th1, 3) if abs(th1) > 1e-6 else np.nan,
                        b_logHR=round(b["coef"], 4), b_p=round(b["p"], 4),
                        mediated_within_RT=round(medprop, 3) if medprop == medprop else np.nan))
        print(f"    mediation: a={a_coef:+.3f} (rho={rho:+.3f}) b={b['coef']:+.3f}(p={b['p']:.2g}) "
              f"inter {th1:+.3f}->{th2:+.3f}  attenu={(th1-th2)/th1 if abs(th1)>1e-6 else float('nan'):.1%}")

pd.DataFrame(rows).to_csv(os.path.join(OUT, "three_cohort_interaction.csv"), index=False)
pd.DataFrame(arms).to_csv(os.path.join(OUT, "three_cohort_arm_HR.csv"), index=False)
pd.DataFrame(med).to_csv(os.path.join(OUT, "three_cohort_mediation.csv"), index=False)

# ------------------------------------------------------------------ pooled (OS)
def pool(df, col_log="logHR", col_se="se"):
    y = df[col_log].values.astype(float); se = df[col_se].values.astype(float)
    w = 1 / se ** 2
    fe = (w * y).sum() / w.sum(); fe_se = np.sqrt(1 / w.sum())
    Q = (w * (y - fe) ** 2).sum(); k = len(y)
    C = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0.0, (Q - (k - 1)) / C) if C > 0 else 0.0
    wr = 1 / (se ** 2 + tau2)
    re = (wr * y).sum() / wr.sum(); re_se = np.sqrt(1 / wr.sum())
    I2 = max(0.0, (Q - (k - 1)) / Q) * 100 if Q > 0 else 0.0
    return dict(k=k, FE_logHR=fe, FE_SE=fe_se, FE_HR=np.exp(fe),
                FE_lo=np.exp(fe - 1.96 * fe_se), FE_hi=np.exp(fe + 1.96 * fe_se),
                FE_p=2 * (1 - norm.cdf(abs(fe / fe_se))),
                RE_HR=np.exp(re), RE_lo=np.exp(re - 1.96 * re_se), RE_hi=np.exp(re + 1.96 * re_se),
                RE_p=2 * (1 - norm.cdf(abs(re / re_se))), Q=Q, I2=I2, tau2=tau2)


it = pd.DataFrame(rows)
os_it = it[it.endpoint == "OS"]
pl = pool(os_it)
print("\n=== pooled OS interaction (fixed effect) ===")
for _, r in os_it.iterrows():
    print(f"  {r.cohort:10s} logHR={r.logHR:+.3f} SE={r.se:.3f} HR={r.HR:.2f} p={r.p:.3g}")
print(f"  POOLED FE: HR={pl['FE_HR']:.3f} ({pl['FE_lo']:.3f}-{pl['FE_hi']:.3f}) p={pl['FE_p']:.4g} "
      f"| RE: HR={pl['RE_HR']:.3f} ({pl['RE_lo']:.3f}-{pl['RE_hi']:.3f}) p={pl['RE_p']:.4g} "
      f"| I2={pl['I2']:.0f}% Q={pl['Q']:.2f}")
pd.DataFrame([pl]).to_csv(os.path.join(OUT, "pooled_OS_interaction.csv"), index=False)

# pooled a-path (Fisher z)
a_rows = pd.DataFrame(med)
if len(a_rows) >= 2:
    z = [np.arctanh(r) for r in a_rows.a_rho]
    n = a_rows.n.values; w = n - 3
    zbar = (w * np.array(z)).sum() / w.sum(); zse = np.sqrt(1 / w.sum())
    print(f"\npooled a-path (DEPTOR->hypoxia): rho={np.tanh(zbar):+.3f} "
          f"95%CI {np.tanh(zbar-1.96*zse):+.3f}..{np.tanh(zbar+1.96*zse):+.3f} "
          f"p={2*(1-norm.cdf(abs(zbar/zse))):.2g}")
    pd.DataFrame([dict(rho=float(np.tanh(zbar)), lo=float(np.tanh(zbar - 1.96 * zse)),
                       hi=float(np.tanh(zbar + 1.96 * zse)),
                       p=float(2 * (1 - norm.cdf(abs(zbar / zse)))), k=len(z))]).to_csv(
        os.path.join(OUT, "pooled_a_path.csv"), index=False)

# ------------------------------------------------------------------ dichotomous (median-split) interaction
# The strong TCGA result reported earlier came from a HIGH-vs-LOW split, not from a
# linear per-SD term.  Refit the same way in all three cohorts and pool, so the
# threshold hypothesis is tested consistently.
drows = []
for name, d in COH.items():
    for endpoint, (t, e) in {"OS": ("os_t", "os_e")}.items():
        dd = d.dropna(subset=[t, e, "DEPTOR", "treat"]).copy()
        if len(dd) < 40: continue
        dd["high"] = (dd["DEPTOR"] > dd["DEPTOR"].median()).astype(int)
        dd["inter"] = dd["high"] * dd["treat"]
        r = fit_cox(dd, ["high", "treat", "inter"], t, e)
        if r is None: continue
        cph, ddd = r; s = cph.summary.loc["inter"]
        drows.append(dict(cohort=name, endpoint=endpoint, logHR=s["coef"], se=s["se(coef)"], p=s["p"],
                          HR=np.exp(s["coef"]), lo=np.exp(s["coef lower 95%"]),
                          hi=np.exp(s["coef upper 95%"]), n=len(ddd), events=int(ddd[e].sum())))
        print(f"[{name}] {endpoint} DICHOTOMOUS interaction HR={np.exp(s['coef']):.2f} "
              f"({np.exp(s['coef lower 95%']):.2f}-{np.exp(s['coef upper 95%']):.2f}) p={s['p']:.3g}")
pd.DataFrame(drows).to_csv(os.path.join(OUT, "three_cohort_interaction_dichotomous.csv"), index=False)
if len(drows) >= 2:
    pl2 = pool(pd.DataFrame(drows))
    print(f"  POOLED (dichotomous) FE: HR={pl2['FE_HR']:.3f} ({pl2['FE_lo']:.3f}-{pl2['FE_hi']:.3f}) "
          f"p={pl2['FE_p']:.4g} | I2={pl2['I2']:.0f}%")
    pd.DataFrame([pl2]).to_csv(os.path.join(OUT, "pooled_OS_interaction_dichotomous.csv"), index=False)

# TCGA HPV-negative subgroup (the pre-specified subgroup)
dT = COH["TCGA-HNSC"]
sub = dT[dT.hpv_neg == True].copy()
sub["high"] = (sub["DEPTOR"] > sub["DEPTOR"].median()).astype(int)
sub["inter"] = sub["high"] * sub["treat"]
for endpoint, (t, e) in {"OS": ("os_t", "os_e"), "DSS": ("dss_t", "dss_e")}.items():
    r = fit_cox(sub, ["high", "treat", "inter"], t, e)
    if r is None: continue
    cph, ddd = r; s = cph.summary.loc["inter"]
    print(f"[TCGA HPV-] {endpoint} dichotomous interaction HR={np.exp(s['coef']):.2f} "
          f"({np.exp(s['coef lower 95%']):.2f}-{np.exp(s['coef upper 95%']):.2f}) p={s['p']:.3g} n={len(ddd)}")

# ------------------------------------------------------------------ forest plot
from matplotlib.ticker import FixedLocator, NullLocator, FixedFormatter

def _forest(labels, hr, lo, hi, colors, fname, xlabel, title, xticks):
    fig, ax = plt.subplots(figsize=(6.6, 0.62 * len(labels) + 1.35))
    y = np.arange(len(labels))[::-1]
    for xi, yi, l_, h_, c in zip(hr, y, lo, hi, colors):
        ax.errorbar(xi, yi, xerr=[[xi - l_], [h_ - xi]], fmt="o", color=c,
                    ecolor=c, elinewidth=2, capsize=4, ms=7)
    ax.axvline(1, ls="--", color="#888", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(xticks))
    ax.xaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in xticks]))
    ax.xaxis.set_minor_locator(NullLocator())
    lo_x = min(min(lo), 1) * 0.92
    hi_x = max(max(hi), 1) * 1.10
    ax.set_xlim(lo_x, hi_x)
    ax.set_ylim(-0.8, len(labels) - 0.2)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_title(title, fontsize=10, pad=10)
    for xi, yi, v in zip(hr, y, hr):
        ax.annotate(f"{v:.2f}", (xi, yi), textcoords="offset points",
                    xytext=(0, 12), ha="center", va="bottom", fontsize=8, color="#222")
    ax.grid(axis="y", ls=":", color="#ddd", lw=.8)
    ax.set_axisbelow(True)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, fname), dpi=200); plt.close()


lab, hr, lo, hi = [], [], [], []
for _, r in os_it.iterrows():
    lab.append(f"{r.cohort} (n={r.n})"); hr.append(r.HR); lo.append(r.lo); hi.append(r.hi)
lab.append(f"POOLED (n={int(os_it.n.sum())})")
hr.append(pl["FE_HR"]); lo.append(pl["FE_lo"]); hi.append(pl["FE_hi"])
_forest(lab, hr, lo, hi, ["#8c96c6"] * (len(lab) - 1) + ["#54278f"],
        "forest_interaction_OS.png",
        "HR for DEPTOR (per SD) × radiotherapy interaction",
        "DEPTOR × radiotherapy interaction on overall survival",
        [0.5, 0.75, 1.0, 1.5, 2.0])

ar = pd.DataFrame(arms)
lab, hr, lo, hi, col = [], [], [], [], []
for name in COH:
    for arm in ["no RT / uni", "RT / multi"]:
        r = ar[(ar.cohort == name) & (ar.arm == arm)]
        if len(r) == 0: continue
        r = r.iloc[0]
        lab.append(f"{name} · {arm}"); hr.append(r.HR); lo.append(r.lo); hi.append(r.hi)
        col.append("#2c7fb8" if arm == "no RT / uni" else "#d95f0e")
_forest(lab, hr, lo, hi, col, "forest_arm_OS.png",
        "HR per SD DEPTOR (overall survival)",
        "DEPTOR and overall survival within treatment arm",
        [0.25, 0.5, 1.0, 2.0])

# dichotomous forest
if len(drows) >= 2:
    dd_ = pd.DataFrame(drows)
    lab, hr, lo, hi = [], [], [], []
    for _, r in dd_.iterrows():
        lab.append(f"{r.cohort} (n={r.n})"); hr.append(r.HR); lo.append(r.lo); hi.append(r.hi)
    lab.append("POOLED"); hr.append(pl2["FE_HR"]); lo.append(pl2["FE_lo"]); hi.append(pl2["FE_hi"])
    _forest(lab, hr, lo, hi, ["#8c96c6"] * (len(lab) - 1) + ["#54278f"],
            "forest_interaction_OS_dichotomous.png",
            "HR for DEPTOR (high vs low) × radiotherapy interaction",
            "Dichotomised DEPTOR × radiotherapy interaction (OS)",
            [0.2, 0.35, 0.5, 1.0, 2.0, 4.0])

print("\nSaved ->", OUT)
