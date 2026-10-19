#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Publication-quality Kaplan-Meier figures with numbers-at-risk tables.

Layout per panel (top -> bottom):
    title  (with panel letter)
    legend (outside, above the axes -> never covers the curves)
    KM axes
    "No. at risk" table aligned to the same x ticks
    x-axis label (time)

Output: data/results_final_figures/
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
import warnings
warnings.filterwarnings("ignore")

BASE = "WorkBuddy/2026-10-17-11-15-46"
OUT = os.path.join(BASE, "data", "results_final_figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "axes.titlesize": 10.0,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.8,
    "ytick.labelsize": 8.8,
    "legend.fontsize": 8.5,
    "axes.linewidth": 0.9,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "figure.dpi": 110,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.12,
})

C_LOW, C_HIGH = "#2c7fb8", "#d94701"          # low / high
C_QUAD = ["#b2182b", "#ef8a62", "#67a9cf", "#2166ac"]

# ----------------------------------------------------------------------------- data
clin = pd.read_csv(os.path.join(BASE, "data", "tcga_hnsc_pca_deptor.csv"))
nk = pd.read_csv(os.path.join(BASE, "data", "tcga_hnsc_nkg2d_nk.csv"))
df = clin.merge(nk, on="sampleId", how="inner")


def ev(series):
    return series.astype(str).str.upper().str.strip().str.startswith("1").astype(int)


df["OS_event"] = ev(df.OS_STATUS)
df["PFS_event"] = ev(df.PFS_STATUS)
df["DSS_event"] = ev(df.DSS_STATUS)
for t in ["OS", "PFS", "DSS"]:
    df[f"{t}_MONTHS"] = pd.to_numeric(df[f"{t}_MONTHS"], errors="coerce")
    df[f"{t}_YEARS"] = df[f"{t}_MONTHS"] / 12.0

df["RT"] = df.RADIATION_THERAPY.astype(str).str.strip()
df["HPV"] = np.where(df.SUBTYPE.astype(str).str.contains("HPV-"), "HPV-",
                     np.where(df.SUBTYPE.astype(str).str.contains(r"HPV\+"), "HPV+", "NA"))
df["DEPTOR"] = pd.to_numeric(df.DEPTOR_RSEM, errors="coerce")

# NKG2D ligand / NK scores (identical to nkg2d_nk_survival.py: mean z-score)
LIG = ["MICA", "MICB", "ULBP1", "ULBP2", "ULBP3", "RAET1E", "RAET1G", "RAET1L"]
NKG = ["NKG7", "GNLY", "KLRD1", "PRF1", "FCGR3A", "NCAM1", "KLRF1", "XCL1", "XCL2", "TBX21", "EOMES"]
for g in LIG + NKG:
    v = pd.to_numeric(df[g], errors="coerce")
    df[g] = v
    df[g + "_z"] = (v - v.mean()) / v.std()
df["NKG2D_ligand"] = df[[g + "_z" for g in LIG]].mean(axis=1)
df["NK_signature"] = df[[g + "_z" for g in NKG]].mean(axis=1)

# global median split (same definition as the original analysis)
df["DEP_HI"] = df.DEPTOR > df.DEPTOR.median()
df["LIG_HI"] = df.NKG2D_ligand > df.NKG2D_ligand.median()
df["NK_HI"] = df.NK_signature > df.NK_signature.median()

# ----------------------------------------------------------------------------- helpers
def binary_stats(d, tcol, ecol, mask_hi):
    """Cox HR (high vs low) + log-rank p."""
    sub = pd.DataFrame({
        "duration": d.loc[mask_hi.index, tcol],
        "event": d.loc[mask_hi.index, ecol],
        "grp": mask_hi.astype(int),
    }).dropna()
    if sub.grp.nunique() < 2 or len(sub) < 10:
        return None
    cph = CoxPHFitter()
    cph.fit(sub, duration_col="duration", event_col="event")
    s = cph.summary.loc["grp"]
    a, b = sub[sub.grp == 1], sub[sub.grp == 0]
    lr = logrank_test(a["duration"], b["duration"], a["event"], b["event"])
    return dict(HR=s["exp(coef)"], lo=s["exp(coef) lower 95%"],
                hi=s["exp(coef) upper 95%"], p_cox=s["p"], p_lr=lr.p_value, n=len(sub))


def stats_text(st, label_hi="high vs low"):
    if st is None:
        return ""
    return (f"DEPTOR {label_hi}\n"
            f"HR {st['HR']:.2f} (95% CI {st['lo']:.2f}-{st['hi']:.2f})\n"
            f"log-rank P = {st['p_lr']:.3g}")


def km_panel(fig, subspec, data, groups, tcol, ecol, title, xlabel, ticks, tmax,
             ylabel=None, stats_txt=None, title_pad=26, legend_ncol=2):
    """One KM panel + numbers-at-risk table."""
    inner = subspec.subgridspec(2, 1, height_ratios=[4.2, 1.32], hspace=0.10)
    ax = fig.add_subplot(inner[0])
    axt = fig.add_subplot(inner[1], sharex=ax)

    kmf = KaplanMeierFitter()
    for label, mask, color in groups:
        T = data.loc[mask, tcol]
        E = data.loc[mask, ecol]
        keep = T.notna() & E.notna()
        T, E = T[keep], E[keep]
        kmf.fit(T, E, label=f"{label} (n={len(T)})")
        kmf.plot_survival_function(ax=ax, ci_show=False, color=color,
                                   linewidth=1.75, show_censors=True,
                                   censor_styles={"ms": 2.6, "marker": "|", "mew": 0.9})

    xpad = 0.045 * tmax
    ax.set_xlim(-xpad, tmax)
    ax.set_ylim(0, 1.03)
    ax.set_yticks([0, 0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["0", "0.25", "0.50", "0.75", "1.00"])
    ax.set_xticks(ticks)
    ax.set_xticklabels([])
    ax.tick_params(axis="x", length=0)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", pad=title_pad, fontsize=10, fontweight="semibold")

    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=legend_ncol,
              frameon=False, handlelength=1.5, columnspacing=1.1,
              borderaxespad=0.0, labelspacing=0.25)

    if stats_txt:
        ax.text(0.025, 0.035, stats_txt, transform=ax.transAxes, va="bottom", ha="left",
                fontsize=8.3, linespacing=1.35, zorder=6,
                bbox=dict(boxstyle="round,pad=0.32", facecolor="white",
                          edgecolor="0.72", linewidth=0.7, alpha=0.96))

    # ---- numbers at risk ----
    n = len(groups)
    axt.set_ylim(0.15, n + 1.38)
    for i, (label, mask, color) in enumerate(groups):
        y = n - i
        T = data.loc[mask, tcol].dropna()
        for t in ticks:
            axt.text(t, y, f"{int((T >= t).sum())}", ha="center", va="center",
                     fontsize=8.4, color="0.12")
    axt.set_yticks([n - i for i in range(n)])
    axt.set_yticklabels([g[0] for g in groups], fontsize=8.3)
    axt.tick_params(axis="y", length=0, pad=12)
    for s in axt.spines.values():
        s.set_visible(False)
    axt.set_xticks(ticks)
    axt.set_xticklabels([f"{t:g}" for t in ticks], fontsize=8.6)
    axt.tick_params(axis="x", length=3.2)
    axt.set_xlabel(xlabel, labelpad=3.5)
    axt.text(-xpad + 0.004 * tmax, n + 0.80, "No. at risk", ha="left", va="center",
             fontsize=7.4, color="0.40", style="italic")
    return ax, axt


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p)
    plt.close(fig)
    print("  saved:", name)


# ============================================================== F1: RT interaction
print("[F1] radiotherapy interaction (OS/PFS/DSS x RT no/yes)")
epi = [("OS", "Overall survival"), ("PFS", "Progression-free survival"), ("DSS", "Disease-specific survival")]
fig = plt.figure(figsize=(10.6, 12.9))
gs = fig.add_gridspec(3, 2, hspace=0.42, wspace=0.32,
                      left=0.105, right=0.985, top=0.905, bottom=0.055)
letter = {"OS": "A", "PFS": "C", "DSS": "E"}
TCK = list(range(0, 15, 2))
for r, (t, lab) in enumerate(epi):
    for c, rt in enumerate(["No", "Yes"]):
        d = df[df.RT == rt].copy()
        mask_hi = d.DEP_HI
        st = binary_stats(d, f"{t}_YEARS", f"{t}_event", mask_hi)
        groups = [("DEPTOR low", ~mask_hi, C_LOW), ("DEPTOR high", mask_hi, C_HIGH)]
        km_panel(fig, gs[r, c], d, groups, f"{t}_YEARS", f"{t}_event",
                 title=f"{letter[t] if c == 0 else chr(ord(letter[t]) + 1)}  Radiation: {rt.lower()}",
                 xlabel="Time (years)", ticks=TCK, tmax=14.2,
                 ylabel=lab if c == 0 else None,
                 stats_txt=stats_text(st).replace("DEPTOR high vs low", "DEPTOR high vs low"),
                 title_pad=28, legend_ncol=2)
fig.suptitle("DEPTOR expression and radiotherapy interaction in TCGA-HNSC (n = 447)",
             fontsize=11.5, fontweight="bold", y=0.985)
save(fig, "F1_KM_radiotherapy_interaction.png")

# ============================================================== F2: all patients
print("[F2] all patients (OS/PFS/DSS)")
fig = plt.figure(figsize=(14.6, 5.6))
gs = fig.add_gridspec(1, 3, wspace=0.26, left=0.055, right=0.99, top=0.79, bottom=0.115)
for c, (t, lab) in enumerate(epi):
    st = binary_stats(df, f"{t}_YEARS", f"{t}_event", df.DEP_HI)
    groups = [("DEPTOR low", ~df.DEP_HI, C_LOW), ("DEPTOR high", df.DEP_HI, C_HIGH)]
    km_panel(fig, gs[0, c], df, groups, f"{t}_YEARS", f"{t}_event",
             title=f"{chr(65 + c)}  All patients", xlabel="Time (years)",
             ticks=list(range(0, 19, 3)), tmax=18.5,
             ylabel="Survival probability" if c == 0 else None,
             stats_txt=stats_text(st), title_pad=28, legend_ncol=1)
fig.suptitle("DEPTOR is not prognostic in unselected HNSCC (TCGA-HNSC, n = 515)",
             fontsize=11.5, fontweight="bold", y=0.985)
save(fig, "F2_KM_all_patients.png")

# ============================================================== F3: HPV stratification
print("[F3] HPV stratification (RT = yes)")
fig = plt.figure(figsize=(10.4, 9.4))
gs = fig.add_gridspec(2, 2, hspace=0.46, wspace=0.30,
                      left=0.095, right=0.985, top=0.875, bottom=0.075)
TCK3 = list(range(0, 15, 2))
letters = [["A", "B"], ["C", "D"]]
for r, (t, lab) in enumerate([("DSS", "Disease-specific survival"), ("PFS", "Progression-free survival")]):
    for c, hpv in enumerate(["HPV-", "HPV+"]):
        d = df[(df.RT == "Yes") & (df.HPV == hpv)].copy()
        st = binary_stats(d, f"{t}_YEARS", f"{t}_event", d.DEP_HI)
        groups = [("DEPTOR low", ~d.DEP_HI, C_LOW), ("DEPTOR high", d.DEP_HI, C_HIGH)]
        km_panel(fig, gs[r, c], d, groups, f"{t}_YEARS", f"{t}_event",
                 title=f"{letters[r][c]}  {hpv}  (n = {len(d)})",
                 xlabel="Time (years)", ticks=TCK3, tmax=14.2,
                 ylabel=lab if c == 0 else None,
                 stats_txt=stats_text(st), title_pad=28, legend_ncol=2)
fig.suptitle("Radiotherapy benefit is confined to HPV-negative HNSCC (irradiated patients)",
             fontsize=11.5, fontweight="bold", y=0.972)
save(fig, "F3_KM_HPV_stratified.png")

# ============================================================== F4: 4 quadrants
print("[F4] NKG2D ligand x NK quadrant (DSS, RT = yes)")
rt = df[df.RT == "Yes"].copy()


def quad(row):
    l, n = row.LIG_HI, row.NK_HI
    if l and n:
        return "ligand-high / NK-high"
    if l and not n:
        return "ligand-high / NK-LOW"
    if not l and n:
        return "ligand-low / NK-high"
    return "ligand-low / NK-low"


df["QUAD"] = df.apply(quad, axis=1)
rt = df[df.RT == "Yes"].copy()
order = ["ligand-high / NK-LOW", "ligand-high / NK-high", "ligand-low / NK-low", "ligand-low / NK-high"]
groups = [(q, rt.QUAD == q, C_QUAD[i]) for i, q in enumerate(order)]
key, rest = rt.QUAD == "ligand-high / NK-LOW", rt.QUAD != "ligand-high / NK-LOW"
lr = logrank_test(rt.loc[key, "DSS_MONTHS"], rt.loc[rest, "DSS_MONTHS"],
                  rt.loc[key, "DSS_event"], rt.loc[rest, "DSS_event"])
mv = multivariate_logrank_test(rt.DSS_MONTHS, rt.QUAD, rt.DSS_event)
quad_stats = (f"'ligand-high / NK-LOW' vs rest\n"
              f"log-rank P = {lr.p_value:.3f}\n"
              f"4-group log-rank P = {mv.p_value:.3f}")
fig = plt.figure(figsize=(7.8, 7.0))
gs = fig.add_gridspec(1, 1, left=0.20, right=0.98, top=0.795, bottom=0.135)
km_panel(fig, gs[0, 0], rt, groups, "DSS_MONTHS", "DSS_event",
         title="A  Irradiated patients, disease-specific survival",
         xlabel="Time (months)", ticks=list(range(0, 169, 24)), tmax=172,
         ylabel="Survival probability", stats_txt=quad_stats, title_pad=44, legend_ncol=2)
fig.suptitle("NKG2D-ligand-high tumours with few NK cells have the poorest long-term outcome",
             fontsize=11, fontweight="bold", y=0.975)
save(fig, "F4_KM_NKG2D_quadrant_DSS.png")

# ============================================================== F5: ligand / NK univariate
print("[F5] NKG2D ligand and NK signature (DSS, RT = yes)")
fig = plt.figure(figsize=(12.2, 6.2))
gs = fig.add_gridspec(1, 2, wspace=0.26, left=0.065, right=0.99, top=0.795, bottom=0.115)
specs = [("NKG2D_ligand", "ligand-high", "ligand-low", "B  NKG2D ligand (DSS)"),
         ("NK_signature", "NK-high", "NK-low", "C  NK signature (DSS)")]
for c, (col, hi_lab, lo_lab, ttl) in enumerate(specs):
    hi = df[col] > df[col].median()
    d = df[df.RT == "Yes"].copy()
    hi_rt = d[col] > d[col].median()
    st = binary_stats(d, "DSS_MONTHS", "DSS_event", hi_rt)
    txt = (f"{hi_lab} vs {lo_lab}\n"
           f"HR {st['HR']:.2f} (95% CI {st['lo']:.2f}-{st['hi']:.2f})\n"
           f"log-rank P = {st['p_lr']:.3g}")
    groups = [(hi_lab, hi_rt, C_HIGH), (lo_lab, ~hi_rt, C_LOW)]
    km_panel(fig, gs[0, c], d, groups, "DSS_MONTHS", "DSS_event", title=ttl,
             xlabel="Time (months)", ticks=list(range(0, 169, 24)), tmax=172,
             ylabel="Survival probability" if c == 0 else None,
             stats_txt=txt, title_pad=28, legend_ncol=2)
fig.suptitle("Each component alone is weaker than their combination (irradiated TCGA-HNSC)",
             fontsize=11.5, fontweight="bold", y=0.975)
save(fig, "F5_KM_NKG2D_univariate.png")

# ============================================================== F6: GSE65858
print("[F6] GSE65858 independent cohort (OS)")
g = pd.read_csv(os.path.join(BASE, "data", "results_deconv65858", "gse65858_deconv_sample.csv"), index_col=0)
g["OS_M"] = pd.to_numeric(g.os, errors="coerce") / 30.4375
g["OS_E"] = pd.to_numeric(g.os_event, errors="coerce").fillna(0).astype(int)
g["DEP"] = pd.to_numeric(g.DEPTOR, errors="coerce")
g = g.dropna(subset=["OS_M", "DEP"])
g_hi = g.DEP > g.DEP.median()
st = binary_stats(g, "OS_M", "OS_E", g_hi)
groups = [("DEPTOR low", ~g_hi, C_LOW), ("DEPTOR high", g_hi, C_HIGH)]
fig = plt.figure(figsize=(7.4, 6.6))
gs = fig.add_gridspec(1, 1, left=0.145, right=0.985, top=0.815, bottom=0.135)
km_panel(fig, gs[0, 0], g, groups, "OS_M", "OS_E",
         title=f"A  GSE65858, overall survival (n = {len(g)})",
         xlabel="Time (months)", ticks=list(range(0, 79, 12)), tmax=80,
         ylabel="Survival probability", stats_txt=stats_text(st), title_pad=28, legend_ncol=2)
fig.suptitle("DEPTOR shows no prognostic main effect in an independent cohort",
             fontsize=11, fontweight="bold", y=0.965)
save(fig, "F6_KM_GSE65858_OS.png")

print("\nDone. Output ->", OUT)
