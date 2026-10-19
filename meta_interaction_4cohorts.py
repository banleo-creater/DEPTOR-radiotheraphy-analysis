#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Meta-analysis of the DEPTOR(high) x radiotherapy interaction across HNSCC cohorts.

Cohorts with a usable RT contrast + survival:
  TCGA-HNSC   : RADIATION_THERAPY Yes/No              (n=447, OS events 165)
  GSE65858    : mono- vs multi-modality (RT proxy)    (n=270, OS events  94)
  GSE41613    : uni-  vs multi-modality (RT proxy)    (n= 96, OS events  50)
  GSE42743    : CEL-only, not processable here        (n=103)  -> excluded

All estimates are the median-split DEPTOR-high x treatment interaction from a
Cox model (log HR + SE), i.e. the same specification as the TCGA discovery.

Outputs -> data/results_meta_rt/
  meta_interaction_estimates.csv   per-cohort logHR / SE / HR / CI / n / events
  meta_interaction_pooled.csv      fixed-effect, DerSimonian-Laird random effects,
                                   Q, I2, tau2, and power to detect the TCGA effect
  F_meta_interaction_forest.png    forest plot
"""
import os
import numpy as np, pandas as pd
from scipy.stats import norm, chi2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = r"WorkBuddy/2026-10-17-11-15-46"
OUT = os.path.join(BASE, "data", "results_meta_rt")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- estimates
# (cohort, endpoint, logHR, se, n, events, rt_definition)
EST = [
    ("TCGA-HNSC", "OS",  -0.9461, 0.3224, 447, 165, "RADIATION_THERAPY Yes/No"),
    ("GSE65858",  "OS",   0.0132, 0.4713, 270,  94, "mono- vs multi-modality"),
    ("GSE41613",  "OS",   0.0770, 0.6191,  96,  50, "uni- vs multi-modality"),
]
df = pd.DataFrame(EST, columns=["cohort", "endpoint", "logHR", "se", "n", "events", "rt_definition"])
df["HR"] = np.exp(df.logHR)
df["lo"] = np.exp(df.logHR - 1.96 * df.se)
df["hi"] = np.exp(df.logHR + 1.96 * df.se)
df["z"] = df.logHR / df.se
df["p"] = 2 * norm.sf(np.abs(df.z))

# ---------------------------------------------------------------- pooling
def pool(df, label):
    y = df.logHR.values; se = df.se.values
    w = 1.0 / se**2
    fe = float(np.sum(w * y) / np.sum(w))
    fe_se = float(np.sqrt(1.0 / np.sum(w)))
    fe_p = float(2 * norm.sf(abs(fe / fe_se)))
    Q = float(np.sum(w * (y - fe)**2))
    k = len(y); dfree = k - 1
    Qp = float(chi2.sf(Q, dfree)) if dfree > 0 else np.nan
    I2 = max(0.0, (Q - dfree) / Q) * 100 if Q > 0 and dfree > 0 else 0.0
    C = float(np.sum(w) - np.sum(w**2) / np.sum(w)) if k > 1 else np.nan
    tau2 = max(0.0, (Q - dfree) / C) if (C and C > 0) else 0.0
    ws = 1.0 / (se**2 + tau2)
    re = float(np.sum(ws * y) / np.sum(ws))
    re_se = float(np.sqrt(1.0 / np.sum(ws)))
    re_p = float(2 * norm.sf(abs(re / re_se)))
    return dict(model=label, k=k,
                FE_logHR=round(fe, 4), FE_SE=round(fe_se, 4), FE_HR=round(float(np.exp(fe)), 3),
                FE_lo=round(float(np.exp(fe - 1.96 * fe_se)), 3),
                FE_hi=round(float(np.exp(fe + 1.96 * fe_se)), 3), FE_p=round(fe_p, 4),
                RE_logHR=round(re, 4), RE_SE=round(re_se, 4), RE_HR=round(float(np.exp(re)), 3),
                RE_lo=round(float(np.exp(re - 1.96 * re_se)), 3),
                RE_hi=round(float(np.exp(re + 1.96 * re_se)), 3), RE_p=round(re_p, 4),
                Q=round(Q, 3), Q_df=dfree, Q_p=round(Qp, 4) if not np.isnan(Qp) else np.nan,
                I2=round(I2, 1), tau2=round(tau2, 4))

pooled = pool(df, "all_3_cohorts")

# ---------------------------------------------------------------- power
# Two-sided alpha=0.05 power to detect the observed TCGA effect (|logHR|)
tcga_loghr = abs(df.loc[df.cohort == "TCGA-HNSC", "logHR"].iloc[0])
z_alpha = norm.ppf(0.975)
pw = []
for _, r in df.iterrows():
    ncp = tcga_loghr / r.se
    power = norm.sf(z_alpha - ncp) + norm.cdf(-z_alpha - ncp)
    pw.append(dict(cohort=r.cohort, n=int(r.n), events=int(r.events),
                   se_logHR=round(float(r.se), 3),
                   min_detectable_HR_at_80pct=round(float(np.exp(2.802 * r.se)), 3),
                   power_to_detect_TCGA_HR=round(float(power), 3)))
power_df = pd.DataFrame(pw)

print("=== per-cohort interaction (DEPTOR-high x RT, OS) ===")
print(df[["cohort", "n", "events", "HR", "lo", "hi", "p", "rt_definition"]].to_string(index=False))
print("\n=== pooled ===")
for k, v in pooled.items():
    print(f"  {k}: {v}")
print("\n=== power to detect the TCGA effect ===")
print(power_df.to_string(index=False))

df.round(4).to_csv(os.path.join(OUT, "meta_interaction_estimates.csv"), index=False)
pd.DataFrame([pooled]).to_csv(os.path.join(OUT, "meta_interaction_pooled.csv"), index=False)
power_df.to_csv(os.path.join(OUT, "meta_interaction_power.csv"), index=False)

# ---------------------------------------------------------------- forest plot
fig, ax = plt.subplots(figsize=(7.2, 3.6))
labels = list(df.cohort) + ["Fixed effect", "Random effects (DL)"]
ys = np.arange(len(labels))[::-1]

for i, r in df.iterrows():
    y = ys[i]
    ax.plot([r.lo, r.hi], [y, y], color="#333333", lw=1.4, zorder=2)
    ax.plot(r.HR, y, "s", color="#1f4e79", ms=7, zorder=3)
    ax.text(4.6, y, f"{r.HR:.2f} ({r.lo:.2f}-{r.hi:.2f})", va="center", fontsize=8)

y_fe, y_re = ys[len(df)], ys[len(df) + 1]
ax.plot([pooled["FE_lo"], pooled["FE_hi"]], [y_fe, y_fe], color="#7f7f7f", lw=1.4)
ax.plot(pooled["FE_HR"], y_fe, "D", color="#7f7f7f", ms=7)
ax.text(4.6, y_fe, f"{pooled['FE_HR']:.2f} ({pooled['FE_lo']:.2f}-{pooled['FE_hi']:.2f})", va="center", fontsize=8)
ax.plot([pooled["RE_lo"], pooled["RE_hi"]], [y_re, y_re], color="#c00000", lw=1.4)
ax.plot(pooled["RE_HR"], y_re, "D", color="#c00000", ms=7)
ax.text(4.6, y_re, f"{pooled['RE_HR']:.2f} ({pooled['RE_lo']:.2f}-{pooled['RE_hi']:.2f})", va="center", fontsize=8)

ax.axvline(1.0, color="black", ls="--", lw=1)
ax.set_xscale("log")
ax.set_xlim(0.08, 6)
ax.set_xticks([0.1, 0.25, 0.5, 1, 2, 4])
ax.set_xticklabels(["0.1", "0.25", "0.5", "1", "2", "4"], fontsize=8)
ax.set_yticks(ys)
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("Interaction HR (DEPTOR-high x radiotherapy)", fontsize=9)
ax.set_title(f"DEPTOR x radiotherapy interaction across HNSCC cohorts\n"
             f"random-effects I²={pooled['I2']:.0f}%, p={pooled['RE_p']:.3f}", fontsize=9)
fig.text(0.5, 0.012, "HR < 1: DEPTOR-high tumours benefit more from radiotherapy",
         ha="center", fontsize=7, style="italic")
ax.grid(axis="x", alpha=0.25, ls=":")
plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig(os.path.join(OUT, "F_meta_interaction_forest.png"), dpi=200)
print(f"\nDONE -> {OUT}")
