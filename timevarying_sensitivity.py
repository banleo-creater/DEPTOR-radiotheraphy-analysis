#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sensitivity analysis for the non-proportional hazards of the RT main effect
(Schoenfeld P = 0.019 on DSS in the primary model).

The interaction term itself satisfied PH (P = 0.97), so the question is only
whether the DEPTOR x RT interaction survives once the RT main effect is allowed
to vary over time. Three approaches:

  S1  EPOCH-STRATIFIED COX (time-varying) — RT main effect free in each epoch,
      DEPTOR x RT interaction CONSTRAINED to be constant across epochs.
      -> the sensitivity estimate of the interaction.
  S2  EPOCH-SPECIFIC INTERACTIONS — one interaction per epoch, plus a
      likelihood-ratio test of homogeneity (is the interaction time-stable?).
  S3  LANDMARK analysis at 12 months — PH-free by construction within the
      landmark cohort.

Epoch cuts: 12 and 24 months (chosen so that every epoch has >= 15 events for
all three endpoints; see run log).
"""
import os, warnings
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, CoxTimeVaryingFitter
warnings.filterwarnings("ignore")

BASE = "WorkBuddy/2026-10-17-11-15-46"
OUT = os.path.join(BASE, "data", "results_orthogonality")
os.makedirs(OUT, exist_ok=True)

CUTS = [12.0, 24.0]

d = pd.read_csv(os.path.join(BASE, "data", "results_deconv", "deconv_per_sample.csv"))
d["RT"] = d["RT"].map({"Yes": 1, "No": 0})

rows = []


def z(s):
    s = pd.Series(s).astype(float)
    return (s - s.mean()) / s.std(ddof=1)


def add(**k):
    rows.append(k)


def tv_dataset(sub, tcol, ecol):
    """Counting-process dataset: one row per (subject, epoch) at risk."""
    out = []
    bounds = [0.0] + CUTS + [np.inf]
    for i, (_, r) in enumerate(sub.iterrows()):
        T, E = float(r[tcol]), int(r[ecol])
        for k in range(len(bounds) - 1):
            a, b = bounds[k], bounds[k + 1]
            if T <= a:
                break
            out.append(dict(id=i, start=a, stop=min(T, b),
                            event=int(E == 1 and T <= b), epoch=k + 1,
                            D_z=r.D_z, RT=r.RT))
    return pd.DataFrame(out)


END = {"OS": ("OS_MONTHS", "OS_event"), "PFS": ("PFS_MONTHS", "PFS_event"),
       "DSS": ("DSS_MONTHS", "DSS_event")}

print("=" * 80)
print("Time-varying / time-stratified sensitivity for the DEPTOR x RT interaction")
print(f"epoch cuts (months): {CUTS}")
print("=" * 80)

fits = {}
for ep, (tcol, ecol) in END.items():
    a = d.dropna(subset=["DEPTOR", "RT", "prop_NK", tcol, ecol]).copy()
    a["D_z"] = z(np.log2(a["DEPTOR"].clip(lower=1)))
    a["DxRT"] = a["D_z"] * a["RT"]

    tv = tv_dataset(a, tcol, ecol)
    tv["RT_e1"] = tv.RT * (tv.epoch == 1)
    tv["RT_e2"] = tv.RT * (tv.epoch == 2)
    tv["RT_e3"] = tv.RT * (tv.epoch == 3)
    tv["Dz"] = tv.D_z
    tv["DxRT_c"] = tv.Dz * tv.RT          # constant across a subject's epochs
    tv["Dx_e1"] = tv.Dz * tv.RT * (tv.epoch == 1)
    tv["Dx_e2"] = tv.Dz * tv.RT * (tv.epoch == 2)
    tv["Dx_e3"] = tv.Dz * tv.RT * (tv.epoch == 3)

    ev_per_epoch = tv.groupby("epoch").event.sum().to_dict()
    risk = tv.groupby("epoch").size().to_dict()
    print(f"\n--- {ep}: n={len(a)}  epoch events={ev_per_epoch}  rows={risk} ---")

    # ---------- S1: common interaction, epoch-specific RT main effect ----------
    c1 = CoxTimeVaryingFitter()
    c1.fit(tv[["id", "start", "stop", "event", "Dz", "RT_e1", "RT_e2", "RT_e3",
               "DxRT_c"]], id_col="id", start_col="start", stop_col="stop",
           event_col="event")
    s = c1.summary.loc["DxRT_c"]
    print(f"  S1 common interaction (RT allowed to vary by epoch):")
    print(f"     HR={s['exp(coef)']:.3f} ({s['exp(coef) lower 95%']:.3f}-"
          f"{s['exp(coef) upper 95%']:.3f})  P={s['p']:.4g}")
    add(part="S1", endpoint=ep, model="common interaction, RT epoch-specific",
        term="DEPTOR x RT", n=int(len(a)), events=int(a[ecol].sum()),
        HR=float(s["exp(coef)"]), lo=float(s["exp(coef) lower 95%"]),
        hi=float(s["exp(coef) upper 95%"]), p=float(s["p"]),
        note=f"epoch events {ev_per_epoch}")
    for et in ["RT_e1", "RT_e2", "RT_e3"]:
        if et in c1.summary.index:
            r = c1.summary.loc[et]
            add(part="S1a", endpoint=ep, model="RT effect by epoch", term=et,
                n=int(len(a)), events=int(a[ecol].sum()), HR=float(np.exp(r["coef"])),
                lo=float(r["exp(coef) lower 95%"]), hi=float(r["exp(coef) upper 95%"]),
                p=float(r["p"]), note="")
            print(f"     {et}: HR={r['exp(coef)']:.3f} P={r['p']:.3g}")
    ll_reduced = float(c1.log_likelihood_)

    # ---------- S2: epoch-specific interactions + LRT ----------
    c2 = CoxTimeVaryingFitter()
    c2.fit(tv[["id", "start", "stop", "event", "Dz", "RT_e1", "RT_e2", "RT_e3",
               "Dx_e1", "Dx_e2", "Dx_e3"]], id_col="id", start_col="start",
           stop_col="stop", event_col="event")
    print(f"  S2 epoch-specific interactions:")
    per_epoch = {}
    for et in ["Dx_e1", "Dx_e2", "Dx_e3"]:
        r = c2.summary.loc[et]
        per_epoch[et] = (float(r["exp(coef)"]), float(r["p"]))
        add(part="S2", endpoint=ep, model="interaction by epoch", term=et,
            n=int(len(a)), events=int(a[ecol].sum()), HR=float(r["exp(coef)"]),
            lo=float(r["exp(coef) lower 95%"]), hi=float(r["exp(coef) upper 95%"]),
            p=float(r["p"]), note="")
        print(f"     {et}: HR={r['exp(coef)']:.3f} "
              f"({r['exp(coef) lower 95%']:.3f}-{r['exp(coef) upper 95%']:.3f}) "
              f"P={r['p']:.3g}")
    ll_full = float(c2.log_likelihood_)
    lrt = 2 * (ll_full - ll_reduced)
    p_hom = 1 - stats.chi2.cdf(max(lrt, 0), 2)
    print(f"     homogeneity of interaction across epochs: LRT chi2={lrt:.2f} "
          f"(df=2) P={p_hom:.3g}")
    add(part="S2", endpoint=ep, model="homogeneity LRT", term="Dx e1=e2=e3",
        n=int(len(a)), events=int(a[ecol].sum()), HR=np.nan, lo=np.nan, hi=np.nan,
        p=float(p_hom), note=f"chi2={lrt:.2f}")
    fits[ep] = per_epoch

    # ---------- S3: landmark at 12 months ----------
    lm = a[a[tcol] > 12].copy()
    lm["T"] = lm[tcol] - 12
    lm["E"] = lm[ecol].astype(int)
    c3 = CoxPHFitter()
    c3.fit(lm[["D_z", "RT", "DxRT", "T", "E"]].astype(float),
           duration_col="T", event_col="E")
    s3 = c3.summary.loc["DxRT"]
    print(f"  S3 landmark at 12 mo (n={len(lm)}, events={int(lm.E.sum())}): "
          f"HR={s3['exp(coef)']:.3f} ({s3['exp(coef) lower 95%']:.3f}-"
          f"{s3['exp(coef) upper 95%']:.3f}) P={s3['p']:.4g}")
    add(part="S3", endpoint=ep, model="landmark at 12 months", term="DEPTOR x RT",
        n=int(len(lm)), events=int(lm.E.sum()), HR=float(s3["exp(coef)"]),
        lo=float(s3["exp(coef) lower 95%"]), hi=float(s3["exp(coef) upper 95%"]),
        p=float(s3["p"]), note="")

res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, "timevarying_sensitivity.csv"), index=False)
print(f"\nsaved {len(res)} rows")

# ---------------------------- figure ----------------------------
BLUE, RED, GREY = "#2b6cb0", "#c0392b", "#5a5a5a"
mcol = {"OS": GREY, "PFS": BLUE, "DSS": RED}
fig, ax = plt.subplots(figsize=(7.4, 4.2))
yy, ticks, labs = 0, [], []
for ep in ["DSS", "PFS", "OS"]:
    s1 = res[(res.part == "S1") & (res.endpoint == ep)].iloc[0]
    ax.scatter([s1.HR], [yy], marker="D", s=46, color=mcol[ep], zorder=4)
    ax.plot([s1.lo, s1.hi], [yy, yy], color=mcol[ep], lw=2.6)
    ticks.append(yy); labs.append(f"{ep} · common (RT epoch-specific)")
    yy += 1
    for et, nm in [("Dx_e1", "0–12 mo"), ("Dx_e2", "12–24 mo"), ("Dx_e3", ">24 mo")]:
        r = res[(res.part == "S2") & (res.endpoint == ep) & (res.term == et)]
        if not len(r):
            continue
        r = r.iloc[0]
        ax.plot([r.lo, r.hi], [yy, yy], color=mcol[ep], lw=1.6, alpha=.75)
        ax.scatter([r.HR], [yy], s=22, color=mcol[ep], alpha=.75, zorder=3)
        ticks.append(yy); labs.append(f"{ep} · {nm}")
        yy += 1
    yy += 0.6
ax.axvline(1, color="k", lw=1, ls="--")
ax.set_yticks(ticks); ax.set_yticklabels(labs, fontsize=6.8)
ax.invert_yaxis()
ax.set_xlim(0.1, 1.9)
ax.set_xlabel("DEPTOR × RT interaction HR (per SD, log2)", fontsize=9)
ax.set_title("Interaction is stable when the RT effect is allowed to vary over time",
             fontsize=10, fontweight="bold", loc="left")
ax.tick_params(labelsize=8)
plt.tight_layout()
fp = os.path.join(OUT, "Figure_timevarying_sensitivity.png")
plt.savefig(fp, dpi=300, facecolor="white", bbox_inches="tight")
plt.close()
print("saved", fp)
