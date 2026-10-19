#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""各队列 DEPTOR x RT 交互 HR 森林图（pooled 2-way），可视化非复制。"""
import os, sys
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import immune_niche_modification as m

OUT = os.path.join(ROOT, "data", "results_epic_quantiseq")
rows = []
for name, loader in m.COHORTS.items():
    y, clin, F = loader()
    imm = m.immune_fraction_from_epic(F)
    dd = clin.join(pd.DataFrame({"DEPTOR": y, "immune": imm}), how="inner")
    dd = dd.dropna(subset=["time", "event", "rt", "DEPTOR", "stage_late"]).copy()
    dd["DEPTOR_hi"] = (dd["DEPTOR"] > dd["DEPTOR"].median()).astype(int)
    dd["d_r"] = dd["DEPTOR_hi"] * dd["rt"]
    c = CoxPHFitter()
    c.fit(dd[["time", "event", "DEPTOR_hi", "rt", "stage_late", "d_r"]],
          duration_col="time", event_col="event")
    b = float(c.params_["d_r"]); se = float(c.summary.loc["d_r", "se(coef)"])
    rows.append(dict(cohort=name, HR=round(np.exp(b), 3),
                     lo=round(np.exp(b - 1.96 * se), 3), hi=round(np.exp(b + 1.96 * se), 3),
                     p=round(float(c.summary.loc["d_r", "p"]), 4), n=len(dd), events=int(dd["event"].sum())))

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "deptor_rt_interaction_forest.csv"), index=False)

fig, ax = plt.subplots(figsize=(7, 3.6))
yp = np.arange(len(df))[::-1]
for i, (_, r) in zip(yp, df.iterrows()):
    col = "#1b9e77" if (r.HR < 1 and r.p < 0.05) else "#999999"
    ax.plot([r.lo, r.hi], [i, i], color=col, lw=2)
    ax.plot(r.HR, i, "o", color=col, ms=7)
    ax.annotate(f"HR={r.HR} ({r.lo}-{r.hi}) p={r.p}", (r.hi, i),
                xytext=(6, 0), textcoords="offset points", fontsize=7, va="center")
ax.axvline(1, color="grey", ls="--", lw=1)
ax.set_yticks(yp); ax.set_yticklabels([f"{r.cohort} (n={r.n}, ev={r.events})" for _, r in df.iterrows()], fontsize=8)
ax.set_xlabel("DEPTOR x radiotherapy interaction HR  (<1: DEPTOR-high tumor benefits more from RT)")
ax.set_title("DEPTOR x RT interaction across HNSCC cohorts (EPIC-calibrated bulk DEPTOR)", fontsize=9)
ax.set_xlim(0, 3.4); ax.grid(axis="x", alpha=0.25, ls=":")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "F_deptor_rt_forest.png"), dpi=130); plt.close()
print(df.to_string(index=False)); print("saved", os.path.join(OUT, "F_deptor_rt_forest.png"))
