#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DEPTOR compartment split x radiotherapy interaction, using EPIC and quanTIseq
(vs the previous in-house NNLS split), in TCGA-HNSC, GSE65858 and GSE41613.

For each cohort and each split (EPIC / quanTIseq / custom) we fit
    Surv(time, event) ~ z(split) * RT  [+ stage]
and report the interaction HR, CI and p.  A *negative* interaction logHR means
higher compartment-attributable DEPTOR goes with greater radiotherapy benefit.

Outputs -> data/results_epic_quantiseq/
  interaction_by_method.csv
  interaction_summary.png
"""
import os, gzip, sys
import numpy as np, pandas as pd
from lifelines import CoxPHFitter
from scipy.stats import chi2 as chi2_dist
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore")

BASE = r"WorkBuddy/2026-10-17-11-15-46"
DATA = os.path.join(BASE, "data")
GEO = r"<PATH>"
SPLIT = os.path.join(DATA, "results_epic_quantiseq")
OUT = SPLIT

# ================================================================ clinical loaders
def tcga_clin():
    d = pd.read_csv(os.path.join(DATA, "tcga_hnsc_pca_deptor.csv"))
    # NOTE: build from .values -- assigning a Series to a frame indexed by
    # sampleId would align on the 0..n RangeIndex and yield all-NaN columns.
    o = pd.DataFrame({
        "patientId": d["patientId"].values,
        "time": pd.to_numeric(d["DSS_MONTHS"], errors="coerce").values,
        "event": d["DSS_STATUS"].astype(str).str.startswith("1").astype(int).values,
        "rt": (d["RADIATION_THERAPY"].astype(str).str.lower() == "yes").astype(int).values,
        "stage_late": d["AJCC_PATHOLOGIC_TUMOR_STAGE"].astype(str).str.contains("III|IV").astype(int).values,
        "endpoint": "DSS",
    }, index=d["sampleId"].values)
    return o


def _chars(path):
    with gzip.open(path, "rt", errors="ignore") as f:
        lines = f.readlines()
    sids, d = None, {}
    for line in lines:
        if line.startswith("!Sample_geo_accession"):
            sids = [v.strip().strip('"') for v in line.rstrip("\n").split("\t")[1:]]
        elif line.startswith("!Sample_characteristics_ch1"):
            vals = [v.strip().strip('"') for v in line.rstrip("\n").split("\t")[1:]]
            if not vals or ":" not in vals[0]:
                continue
            key = vals[0].split(":")[0].strip()
            d[key] = [v.split(":", 1)[-1].strip() for v in vals]
    return pd.DataFrame(d, index=sids)


def gse65858_clin():
    c = _chars(os.path.join(GEO, "GSE65858", "GSE65858_series_matrix.txt.gz"))
    st = c.get("uicc_stage", pd.Series(index=c.index, dtype=str)).astype(str)
    # os_event is stored as a logical ("TRUE"/"FALSE"); os is in days
    ev_raw = c.get("os_event", pd.Series(index=c.index, dtype=str)).astype(str).str.upper()
    ev = ev_raw.map({"TRUE": 1, "FALSE": 0})
    ev = ev.fillna(pd.to_numeric(ev_raw, errors="coerce"))
    t_raw = pd.to_numeric(c.get("os"), errors="coerce")
    t_months = t_raw / 30.4375 if t_raw.median() > 200 else t_raw   # days -> months
    o = pd.DataFrame({
        "time": t_months.values,
        "event": ev.values,
        "rt": c["treatment"].astype(str).str.lower().str.startswith("multi").astype(int).values,
        "stage_late": st.str.contains("III|IV").astype(int).values,
        "endpoint": "OS",
    }, index=c.index)
    return o


def gse41613_clin():
    c = pd.read_csv(os.path.join(DATA, "gse41613_clinical.csv"), index_col=0)
    rt = c["treatment"].astype(str).str.lower().str.contains("multi").astype(int)
    rt[c["treatment"].astype(str).str.lower().eq("unknown")] = np.nan
    o = pd.DataFrame({
        "time": pd.to_numeric(c["fu time"], errors="coerce").values,
        "event": c["vital"].astype(str).str.lower().str.startswith("dead").astype(int).values,
        "rt": rt.values,
        "stage_late": c["tumor stage"].astype(str).str.contains("III|IV").astype(int).values,
        "endpoint": "OS",
    }, index=c.index)
    return o


CLIN = {"TCGA": tcga_clin, "GSE65858": gse65858_clin, "GSE41613": gse41613_clin}

# ================================================================ custom solver splits
def custom_split(name):
    if name == "TCGA":
        p = os.path.join(DATA, "results_deconv", "deconv_per_sample.csv")
        d = pd.read_csv(p, index_col="sampleId")
    elif name == "GSE65858":
        p = os.path.join(DATA, "results_deconv65858", "gse65858_deconv_sample.csv")
        d = pd.read_csv(p, index_col=0)
    else:
        p = os.path.join(DATA, "results_deconv41613", "GSE41613_deconv_sample.csv")
        d = pd.read_csv(p, index_col=0)
    if "DEPTOR_tumor_frac" not in d.columns:
        return None
    return d[["DEPTOR_tumor_frac", "DEPTOR_immune_frac"]].add_prefix("custom_")


# ================================================================ interaction fit
def interaction(d, var, cov_extra=None):
    dd = d.dropna(subset=["time", "event", "rt", var]).copy()
    if len(dd) < 40 or dd.event.sum() < 15:
        return None
    dd["vz"] = (dd[var] - dd[var].mean()) / dd[var].std()
    dd["inter"] = dd["vz"] * dd["rt"]
    covs = ["vz", "rt", "inter"] + (cov_extra or [])
    cph = CoxPHFitter()
    cph.fit(dd[["time", "event"] + covs], duration_col="time", event_col="event")
    s = cph.summary.loc["inter"]
    # LRT
    c0 = CoxPHFitter(); c0.fit(dd[["time", "event", "vz", "rt"] + (cov_extra or [])],
                               duration_col="time", event_col="event")
    ll = max(0.0, 2 * (float(cph.log_likelihood_) - float(c0.log_likelihood_)))
    return dict(n=len(dd), events=int(dd.event.sum()),
                inter_HR=float(np.exp(s["coef"])),
                lo=float(np.exp(s["coef lower 95%"])), hi=float(np.exp(s["coef upper 95%"])),
                p=float(s["p"]), logHR=float(s["coef"]), se=float(s["se(coef)"]),
                LRT_p=float(chi2_dist.sf(ll, 1)),
                sd_var=float(dd[var].std()))


# ================================================================ run
rows = []
for name, loader in CLIN.items():
    clin = loader()
    ep = pd.read_csv(os.path.join(SPLIT, f"{name}_EPIC_DEPTOR_split.csv"), index_col=0)
    qt = pd.read_csv(os.path.join(SPLIT, f"{name}_quantiseq_DEPTOR_split.csv"), index_col=0)
    cu = custom_split(name)

    d = clin.join(ep.add_prefix("EPIC_"), how="inner").join(qt.add_prefix("qT_"), how="inner")
    if cu is not None:
        d = d.join(cu, how="left")

    for method, pref in [("EPIC", "EPIC_"), ("quanTIseq", "qT_"), ("custom", "custom_")]:
        for base in ["DEPTOR_tumor_frac", "DEPTOR_immune_frac", "DEPTOR_myeloid_frac"]:
            var = pref + base
            if var not in d.columns:
                continue
            for adj, tag in [(None, "unadj"), (["stage_late"], "adj_stage")]:
                r = interaction(d, var, adj)
                if r is None:
                    continue
                rows.append(dict(cohort=name, endpoint=clin["endpoint"].iloc[0],
                                 method=method, split=base, model=tag,
                                 **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}))

res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, "interaction_by_method.csv"), index=False)

print("=== interaction by method (adjusted for stage) ===")
show = res[(res.model == "adj_stage") & (res.split != "DEPTOR_myeloid_frac")]
print(show[["cohort", "method", "split", "n", "events", "inter_HR", "lo", "hi", "p"]].to_string(index=False))

# ---------------------------------------------------------------- plot
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
cohorts = list(CLIN.keys())
methods = ["EPIC", "quanTIseq", "custom"]
colors = {"EPIC": "#1f4e79", "quanTIseq": "#c00000", "custom": "#7f7f7f"}
splits = ["DEPTOR_tumor_frac", "DEPTOR_immune_frac"]
offsets = {m: (i - 1) * 0.22 for i, m in enumerate(methods)}

for ax, coh in zip(axes, cohorts):
    sub = res[(res.cohort == coh) & (res.model == "adj_stage")]
    y = 1
    for sp_i, sp in enumerate(splits):
        yy = y - sp_i * 1.6
        for m in methods:
            r = sub[(sub.method == m) & (sub.split == sp)]
            if r.empty:
                continue
            r = r.iloc[0]
            ax.plot([r.lo, r.hi], [yy + offsets[m]] * 2, color=colors[m], lw=1.5)
            ax.plot(r.inter_HR, yy + offsets[m], "o", color=colors[m], ms=6)
    ax.axvline(1, color="k", ls="--", lw=1)
    ax.set_xscale("log"); ax.set_xlim(0.2, 5)
    ax.set_yticks([1, 1 - 1.6])
    ax.set_yticklabels(["tumour\nDEPTOR", "immune\nDEPTOR"])
    ax.set_title(coh, fontsize=10)
    ax.set_xlabel("interaction HR (per SD x RT)", fontsize=8)
    ax.grid(axis="x", alpha=0.25, ls=":")
    ax.tick_params(labelsize=8)

from matplotlib.lines import Line2D
handles = [Line2D([0], [0], color=colors[m], marker="o", lw=1.5, label=m) for m in methods]
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False)
fig.suptitle("DEPTOR compartment split x radiotherapy (stage-adjusted)", fontsize=10)
plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig(os.path.join(OUT, "interaction_summary.png"), dpi=200)
print(f"\nDONE -> {OUT}")
