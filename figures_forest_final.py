#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Clean publication forest plots (log x-axis with fixed ticks, HR/CI/p in a
dedicated right-hand column so nothing overlaps).

Outputs -> data/results_final_figures/
    F7_forest_radiotherapy_interaction.png
    F8_forest_sensitivity_cutpoint.png
    F9_forest_HPV_stratified.png
    forest_stats.csv   (every estimate used, for the manuscript)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
from lifelines import CoxPHFitter
import warnings
warnings.filterwarnings("ignore")

BASE = "WorkBuddy/2026-10-17-11-15-46"
OUT = os.path.join(BASE, "data", "results_final_figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9.5,
    "axes.titlesize": 10.5, "axes.labelsize": 9.5,
    "xtick.labelsize": 8.8, "ytick.labelsize": 9.0,
    "axes.linewidth": 0.9, "figure.dpi": 110,
    "savefig.dpi": 400, "savefig.bbox": "tight", "savefig.pad_inches": 0.12,
})

C_PROT, C_RISK, C_GREY = "#2c7fb8", "#d94701", "#4d4d4d"

# ------------------------------------------------------------------ data
clin = pd.read_csv(os.path.join(BASE, "data", "tcga_hnsc_pca_deptor.csv"))
df = clin.copy()


def ev(s):
    return s.astype(str).str.upper().str.strip().str.startswith("1").astype(int)


df["OS_event"] = ev(df.OS_STATUS)
df["PFS_event"] = ev(df.PFS_STATUS)
df["DSS_event"] = ev(df.DSS_STATUS)
for t in ["OS", "PFS", "DSS"]:
    df[f"{t}_MONTHS"] = pd.to_numeric(df[f"{t}_MONTHS"], errors="coerce")
df["RT"] = df.RADIATION_THERAPY.astype(str).str.strip()
df["HPV"] = np.where(df.SUBTYPE.astype(str).str.contains("HPV-"), "HPV-",
                     np.where(df.SUBTYPE.astype(str).str.contains(r"HPV\+"), "HPV+", "NA"))
df["DEPTOR"] = pd.to_numeric(df.DEPTOR_RSEM, errors="coerce")
CUT_MEDIAN = float(df.DEPTOR.median())
df["DEP_HI"] = df.DEPTOR > CUT_MEDIAN


def cox(d, t, hi):
    """HR for hi vs low; returns dict or None if not estimable."""
    sub = pd.DataFrame({"dur": d.loc[hi.index, f"{t}_MONTHS"],
                        "evt": d.loc[hi.index, f"{t}_event"],
                        "grp": hi.astype(int)}).dropna()
    if sub.grp.nunique() < 2 or len(sub) < 12:
        return None
    if sub.groupby("grp").evt.sum().min() < 2:      # too few events -> unstable
        return None
    try:
        cph = CoxPHFitter()
        cph.fit(sub, duration_col="dur", event_col="evt")
        s = cph.summary.loc["grp"]
        return dict(HR=float(s["exp(coef)"]), lo=float(s["exp(coef) lower 95%"]),
                    hi=float(s["exp(coef) upper 95%"]), p=float(s["p"]), n=len(sub))
    except Exception:
        return None


def draw_forest(rows, title, fname, xlim=(0.18, 5.5), ticks=(0.25, 0.5, 1, 2, 4),
                group_gap=0.9, figsize=(9.6, None)):
    """rows: list of dicts {label, HR, lo, hi, p, n, color, group}"""
    n = len(rows)
    fig_h = figsize[1] if figsize[1] else max(3.2, 0.40 * n + 1.5)
    fig, ax = plt.subplots(figsize=(figsize[0], fig_h))

    # y positions with gaps between groups
    ys, y = [], 0.0
    last_group = None
    for r in rows:
        if last_group is not None and r["group"] != last_group:
            y += group_gap
        ys.append(y)
        last_group = r["group"]
        y += 1.0

    for yy, r in zip(ys, rows):
        ax.plot([r["lo"], r["hi"]], [yy, yy], color=r["color"], lw=1.25,
                solid_capstyle="butt", zorder=3)
        ax.plot([r["lo"], r["lo"]], [yy - .12, yy + .12], color=r["color"], lw=1.25, zorder=3)
        ax.plot([r["hi"], r["hi"]], [yy - .12, yy + .12], color=r["color"], lw=1.25, zorder=3)
        if r["HR"] < xlim[0] or r["HR"] > xlim[1]:      # point off-scale
            ax.annotate("", xy=(xlim[1] * 0.97, yy), xytext=(xlim[1] * 0.85, yy),
                        arrowprops=dict(arrowstyle="->", color=r["color"], lw=1.0))
        else:
            ax.plot([r["HR"]], [yy], "o", ms=6.2, color=r["color"],
                    markeredgecolor="white", markeredgewidth=0.8, zorder=4)

        # right-hand text column (outside the axes -> never overlaps)
        txt = f"{r['HR']:.2f} ({r['lo']:.2f}-{r['hi']:.2f})" if r.get("show") else r.get("note", "")
        pv = r.get("ptext", "")
        ax.text(1.025, yy, txt, transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=8.3, color="0.12")
        ax.text(1.30, yy, pv, transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=8.3, color="0.12")

    ax.axvline(1.0, ls="--", lw=1.0, color="0.45", zorder=1)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(list(ticks)))
    ax.xaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in ticks]))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.set_xlim(*xlim)
    ax.set_ylim(-1.0, max(ys) + 1.0)
    ax.invert_yaxis()

    ax.set_yticks(ys)
    ax.set_yticklabels([r["label"] for r in rows])
    ax.tick_params(axis="y", length=0)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.set_xlabel("Hazard ratio (DEPTOR high vs low, log scale)")
    ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", pad=10)

    # column headers
    ax.text(1.025, -0.85, "HR (95% CI)", transform=ax.get_yaxis_transform(),
            ha="left", va="center", fontsize=7.8, color="0.35", style="italic")
    ax.text(1.30, -0.85, "P", transform=ax.get_yaxis_transform(),
            ha="left", va="center", fontsize=7.8, color="0.35", style="italic")
    fig.subplots_adjust(left=0.30, right=0.72, top=0.90, bottom=0.13)
    fig.savefig(os.path.join(OUT, fname))
    plt.close(fig)
    print("  saved:", fname)


def pfmt(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    return "<0.001" if p < 0.001 else f"{p:.3f}"


# ==================================================== F7 main interaction forest
print("[F7] main forest")
rows = []
for t, tname in [("OS", "OS"), ("PFS", "PFS"), ("DSS", "DSS")]:
    for stratum, d in [("All", df), ("RT no", df[df.RT == "No"]), ("RT yes", df[df.RT == "Yes"])]:
        st = cox(d, t, d.DEP_HI)
        if st is None:
            continue
        rows.append(dict(label=f"{tname}  |  {stratum}  (n={st['n']})", HR=st["HR"], lo=st["lo"],
                         hi=st["hi"], p=st["p"], n=st["n"], show=True, ptext=pfmt(st["p"]),
                         color=C_PROT if st["HR"] < 1 else C_RISK, group=t))
draw_forest(rows,
            "DEPTOR-high vs DEPTOR-low: hazard ratios by endpoint and radiotherapy stratum\n"
            "TCGA-HNSC, dichotomised at the cohort median",
            "F7_forest_radiotherapy_interaction.png",
            xlim=(0.20, 6.0), ticks=(0.25, 0.5, 1, 2, 4), group_gap=0.75,
            figsize=(9.8, 5.6))

# ==================================================== F8 sensitivity forest
print("[F8] sensitivity forest")
sens = pd.read_csv(os.path.join(BASE, "data", "results_tcga", "sens_optimal_vs_median.csv"))
rows = []
for _, r in sens.iterrows():
    t, stratum = r["endpoint"], r["stratum"]
    if stratum == "All":
        d = df
    elif stratum == "RT No":
        d = df[df.RT == "No"]
    else:
        d = df[df.RT == "Yes"]
    for tag, cut, color in [("median", CUT_MEDIAN, C_GREY),
                            ("optimal", float(r["opt_cutoff_RSEM"]), C_PROT)]:
        sub = d.dropna(subset=[f"{t}_MONTHS", "DEPTOR"])
        st = cox(sub, t, (sub.DEPTOR > cut))
        rows.append(dict(label=f"{t} | {stratum}  [{tag} {cut:.1f}]",
                         HR=st["HR"] if st else np.nan, lo=st["lo"] if st else np.nan,
                         hi=st["hi"] if st else np.nan, p=st["p"] if st else np.nan,
                         show=st is not None,
                         note="" if st else "not estimable",
                         ptext=pfmt(st["p"]) if st else "",
                         color=color if st else "0.7", group=f"{t}-{stratum}"))
draw_forest(rows,
            "Sensitivity of the DEPTOR estimate to the cut-point definition\n"
            "grey = cohort median (pre-specified); blue = data-driven optimal cut-point",
            "F8_forest_sensitivity_cutpoint.png",
            xlim=(0.20, 6.0), ticks=(0.25, 0.5, 1, 2, 4), group_gap=0.35,
            figsize=(9.8, 8.6))

# ==================================================== F9 HPV forest (global cut)
print("[F9] HPV forest")
rows = []
hpv_stats = []
for hpv in ["HPV-", "HPV+"]:
    for t in ["OS", "PFS", "DSS"]:
        for stratum in ["All", "RT no", "RT yes"]:
            d = df[(df.HPV == hpv)]
            if stratum == "RT no":
                d = d[d.RT == "No"]
            elif stratum == "RT yes":
                d = d[d.RT == "Yes"]
            st = cox(d, t, d.DEP_HI)
            label = f"{hpv}  {t}  |  {stratum}  (n={len(d)})"
            if st is None:
                rows.append(dict(label=label, HR=np.nan, lo=np.nan, hi=np.nan, p=np.nan,
                                 show=False, note="not estimable", ptext="",
                                 color="0.7", group=hpv))
                hpv_stats.append(dict(HPV=hpv, endpoint=t, stratum=stratum, n=len(d),
                                      HR=np.nan, lo=np.nan, hi=np.nan, p=np.nan))
                continue
            rows.append(dict(label=label, HR=st["HR"], lo=st["lo"], hi=st["hi"], p=st["p"],
                             show=True, ptext=pfmt(st["p"]),
                             color=C_PROT if st["HR"] < 1 else C_RISK, group=hpv))
            hpv_stats.append(dict(HPV=hpv, endpoint=t, stratum=stratum, n=st["n"],
                                  HR=st["HR"], lo=st["lo"], hi=st["hi"], p=st["p"]))
draw_forest(rows,
            "DEPTOR hazard ratios by HPV status (cohort median cut-point, all strata)\n"
            "the radiotherapy benefit is visible only in HPV-negative tumours",
            "F9_forest_HPV_stratified.png",
            xlim=(0.20, 6.0), ticks=(0.25, 0.5, 1, 2, 4), group_gap=0.9,
            figsize=(9.8, 7.4))

# ---------------------------------------------------------------- save table
allrows = []
for r in rows:
    allrows.append(dict(panel="F9_HPV", label=r["label"], HR=r["HR"], lo=r["lo"], hi=r["hi"], p=r["p"]))
pd.DataFrame(hpv_stats).to_csv(os.path.join(OUT, "forest_stats_HPV.csv"), index=False)

# ---------------- interaction terms (global cut) ----------------
print("[TAB] interaction terms (global median cut)")
rows_i = []
dfrt = df[df.RT.isin(["Yes", "No"])]
for scope, d in [("All", dfrt), ("HPV-", dfrt[dfrt.HPV == "HPV-"]), ("HPV+", dfrt[dfrt.HPV == "HPV+"])]:
    for t in ["OS", "PFS", "DSS"]:
        sub = pd.DataFrame({"dur": d[f"{t}_MONTHS"], "evt": d[f"{t}_event"],
                            "hi": d.DEP_HI.astype(int),
                            "rt": (d.RT == "Yes").astype(int)}).dropna()
        if sub.rt.nunique() < 2 or len(sub) < 30:
            continue
        sub["inter"] = sub.hi * sub.rt
        try:
            cph = CoxPHFitter()
            cph.fit(sub, duration_col="dur", event_col="evt")
            s2 = cph.summary.loc["inter"]
            rows_i.append(dict(scope=scope, endpoint=t, n=len(sub),
                               HR_inter=float(s2["exp(coef)"]),
                               lo=float(s2["exp(coef) lower 95%"]),
                               hi=float(s2["exp(coef) upper 95%"]),
                               p=float(s2["p"])))
        except Exception as e:
            rows_i.append(dict(scope=scope, endpoint=t, n=len(sub), HR_inter=np.nan,
                               lo=np.nan, hi=np.nan, p=np.nan))
pd.DataFrame(rows_i).to_csv(os.path.join(OUT, "forest_stats_interaction.csv"), index=False)
print(pd.DataFrame(rows_i).to_string(index=False))
print("  saved: forest_stats_HPV.csv")
print("\nDone ->", OUT)
