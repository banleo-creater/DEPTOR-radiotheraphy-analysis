#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
肿瘤 vs 免疫 DEPTOR —— 经验回归拆分（定量版）
============================================
固定单细胞 centroid 先验会把 DEPTOR 永远判为恶性主导（先验 DEPTOR_malig=0.0413
vs 免疫 0.001~0.0126），使"免疫分室"恒≈0、跨方法无信号。本脚本改为：

  用 EPIC / quanTIseq 的【可靠细胞比例】作设计矩阵，
  从【真实 bulk DEPTOR 表达】经验回归各分室 DEPTOR 表达系数 β，
  再按 β×细胞比例 归因到 肿瘤 / 免疫 / 其他(间质+内皮) 三室。

这样"免疫 DEPTOR"是数据驱动估计，而非先验假设；同时报告 β 与 R²，
可直接判断 DEPTOR 究竟是否主要为恶性表达（β_tumor >> β_immune）。

输出（data/results_epic_quantiseq/）:
  <COHORT>_<METHOD>_DEPTOR_split_empirical.csv
  empirical_compartment_beta.csv     (β_Tumor/β_Immune/β_Other, R2, n)
  empirical_interaction.csv          (肿瘤/免疫 DEPTOR 分室 × RT 交互, 调整分期)
"""
import os, sys, gzip
import numpy as np
import pandas as pd
from scipy import stats
from lifelines import CoxPHFitter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import epic_quantiseq_deconv as eq

DATA = os.path.join(ROOT, "data")
OUT = os.path.join(DATA, "results_epic_quantiseq")
GEO = r"<PATH>"
os.makedirs(OUT, exist_ok=True)


def cox_fit(d, covs, t="time", e="event"):
    cph = CoxPHFitter()
    cph.fit(d[["time", "event"] + covs], duration_col="time", event_col="event")
    return cph, d


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


# ---- 三室聚合映射 ----
def to_three(F, mapping):
    """F: samples x cell-types -> 三室 (Tumor, Immune, Other) 比例和。"""
    Tumor = pd.Series(0.0, index=F.index)
    Immune = pd.Series(0.0, index=F.index)
    Other = pd.Series(0.0, index=F.index)
    inv = {}
    for cell, comp in mapping.items():
        inv.setdefault(comp, []).append(cell)
    for comp, cells in inv.items():
        s = F[[c for c in cells if c in F.columns]].sum(axis=1)
        if comp == "Malignant":
            Tumor = Tumor + s
        elif comp in ("Tcell", "NK", "Bcell", "Myeloid"):
            Immune = Immune + s
        else:  # Stroma, Endo
            Other = Other + s
    return pd.DataFrame({"Tumor": Tumor, "Immune": Immune, "Other": Other}, index=F.index)


# ---- 经验回归：bulk DEPTOR ~ 三室比例(+截距) ----
def empirical_split(y, F3):
    df = pd.concat([y.rename("y"), F3], axis=1).dropna()
    X = np.column_stack([np.ones(len(df)), df["Tumor"].values,
                         df["Immune"].values, df["Other"].values])
    yy = df["y"].values
    beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
    pred = X @ beta
    ss_res = np.sum((yy - pred) ** 2)
    ss_tot = np.sum((yy - yy.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    a, bT, bI, bO = beta
    fT, fI, fO = df["Tumor"].values, df["Immune"].values, df["Other"].values
    # 归因: 每室贡献 = f_c*(a + b_c)，截距按 f_c 比例分配
    contrib_T = fT * (a + bT)
    contrib_I = fI * (a + bI)
    contrib_O = fO * (a + bO)
    tot = contrib_T + contrib_I + contrib_O
    out = pd.DataFrame({
        "DEPTOR_tumor_frac": contrib_T / tot,
        "DEPTOR_immune_frac": contrib_I / tot,
        "DEPTOR_other_frac": contrib_O / tot,
        "beta_Tumor": bT, "beta_Immune": bI, "beta_Other": bO, "intercept": a,
    }, index=df.index)
    return out, r2, len(df)


# ---- 队列 bulk DEPTOR + 临床 ----
def tcga():
    d = pd.read_csv(os.path.join(DATA, "tcga_hnsc_pca_deptor.csv"))
    y = pd.Series(d["DEPTOR_RSEM"].values, index=d["sampleId"].astype(str))
    clin = pd.DataFrame(index=d["sampleId"].astype(str))
    clin["time"] = pd.to_numeric(d["DSS_MONTHS"], errors="coerce").values
    clin["event"] = d["DSS_STATUS"].astype(str).str.startswith("1").astype(int).values
    clin["rt"] = (d["RADIATION_THERAPY"].astype(str).str.lower() == "yes").astype(int).values
    clin["stage_late"] = d["AJCC_PATHOLOGIC_TUMOR_STAGE"].astype(str).str.contains("III|IV").astype(int).values
    clin["endpoint"] = "DSS"
    return y, clin


def gse65858():
    raw = pd.read_csv(os.path.join(DATA, "results_deconv65858", "gse65858_deconv_sample.csv"), index_col=0)
    y = pd.Series(raw["DEPTOR"].values, index=raw.index.astype(str))
    c = _chars(os.path.join(GEO, "GSE65858", "GSE65858_series_matrix.txt.gz"))
    clin = pd.DataFrame(index=c.index.astype(str))
    clin["time"] = (pd.to_numeric(c.get("os"), errors="coerce") / 30.44).values
    clin["event"] = c["os_event"].astype(str).str.upper().str.startswith("T").astype(int).values
    clin["rt"] = c["treatment"].astype(str).str.lower().str.startswith("multi").astype(int).values
    st = c.get("uicc_stage", pd.Series(index=c.index, dtype=str)).astype(str)
    clin["stage_late"] = st.str.contains("III|IV").astype(int).values
    clin["endpoint"] = "OS"
    return y, clin


def gse41613():
    raw = pd.read_csv(os.path.join(DATA, "results_deconv41613", "GSE41613_deconv_sample.csv"), index_col=0)
    y = pd.Series(raw["DEPTOR"].values, index=raw.index.astype(str))
    c = pd.read_csv(os.path.join(DATA, "gse41613_clinical.csv"), index_col=0)
    c.index = c.index.astype(str)
    clin = pd.DataFrame(index=c.index)
    clin["time"] = pd.to_numeric(c["fu time"], errors="coerce").values
    clin["event"] = c["vital"].astype(str).str.lower().str.startswith("dead").astype(int).values
    clin["rt"] = c["treatment"].astype(str).str.lower().str.contains("multi").astype(int).values
    clin.loc[c["treatment"].astype(str).str.lower().eq("unknown"), "rt"] = np.nan
    clin["stage_late"] = c["tumor stage"].astype(str).str.contains("III|IV").astype(int).values
    clin["endpoint"] = "OS"
    return y, clin


COHORTS = {
    "TCGA": (tcga, "TCGA_full_EPIC_fractions.csv", "TCGA_full_quantiseq_fractions.csv",
             eq.EPIC_MAP, eq.QS_MAP),
    "GSE65858": (gse65858, "GSE65858_EPIC_fractions.csv", "GSE65858_quantiseq_fractions.csv",
                 eq.EPIC_MAP, eq.QS_MAP),
    "GSE41613": (gse41613, "GSE41613_EPIC_fractions.csv", "GSE41613_quantiseq_fractions.csv",
                 eq.EPIC_MAP, eq.QS_MAP),
}


def cox_interaction(dd, col):
    dd = dd.dropna(subset=["time", "event", "rt", col, "stage_late"]).copy()
    dd[col + "_hi"] = (dd[col] > dd[col].median()).astype(int)
    dd["inter"] = dd[col + "_hi"] * dd["rt"]
    c0, _ = cox_fit(dd, ["rt", col + "_hi", "stage_late"])
    c1, _ = cox_fit(dd, ["rt", col + "_hi", "stage_late", "inter"])
    lr = 2 * (c1.log_likelihood_ - c0.log_likelihood_)
    p = stats.chi2.sf(lr, 1)
    # 分臂
    rows = []
    for rv in (0, 1):
        s = dd[dd["rt"] == rv]
        if s["event"].sum() < 5 or len(s) < 10:
            continue
        cx, _ = cox_fit(s, [col + "_hi", "stage_late"])
        hr = float(np.exp(cx.params_[col + "_hi"]))
        lo = float(np.exp(cx.params_[col + "_hi"] - 1.96 * cx.summary.loc[col + "_hi", "se(coef)"]))
        hi = float(np.exp(cx.params_[col + "_hi"] + 1.96 * cx.summary.loc[col + "_hi", "se(coef)"]))
        pp = float(cx.summary.loc[col + "_hi", "p"])
        rows.append(dict(rt=rv, HR=round(hr, 3), lo=round(lo, 3), hi=round(hi, 3), p=round(pp, 4),
                         n=len(s), events=int(s["event"].sum())))
    return dict(HR=round(float(np.exp(c1.params_["inter"])), 3),
                lo=round(float(np.exp(c1.params_["inter"] - 1.96 * c1.summary.loc["inter", "se(coef)"])), 3),
                hi=round(float(np.exp(c1.params_["inter"] + 1.96 * c1.summary.loc["inter", "se(coef)"])), 3),
                p=round(p, 4), LRT_chi2=round(float(lr), 2),
                n=len(dd), events=int(dd["event"].sum())), rows


def main():
    beta_rows, inter_rows = [], []
    for name, (loader, fe, fq, emap, qmap) in COHORTS.items():
        y, clin = loader()
        Fe = pd.read_csv(os.path.join(OUT, fe), index_col=0)
        Fq = pd.read_csv(os.path.join(OUT, fq), index_col=0)
        for method, F, mp in [("EPIC", Fe, emap), ("quanTIseq", Fq, qmap)]:
            F3 = to_three(F, mp)
            split, r2, nfit = empirical_split(y, F3)
            split.to_csv(os.path.join(OUT, f"{name}_{method}_DEPTOR_split_empirical.csv"))
            beta_rows.append(dict(cohort=name, method=method, n=nfit, R2=round(r2, 3),
                                  beta_Tumor=round(float(split["beta_Tumor"].median()), 3),
                                  beta_Immune=round(float(split["beta_Immune"].median()), 3),
                                  beta_Other=round(float(split["beta_Other"].median()), 3)))
            # 交互
            dd = clin.join(split[["DEPTOR_tumor_frac", "DEPTOR_immune_frac"]], how="inner")
            for split_col in ["DEPTOR_tumor_frac", "DEPTOR_immune_frac"]:
                res, arm = cox_interaction(dd, split_col)
                inter_rows.append(dict(cohort=name, method=method, split=split_col, **res))
                for a in arm:
                    inter_rows.append(dict(cohort=name, method=method, split=split_col + f"_armRT{a['rt']}",
                                           HR=a["HR"], lo=a["lo"], hi=a["hi"], p=a["p"], n=a["n"], events=a["events"]))
            print(f"  {name:9s} {method:9s} R2={r2:.3f}  "
                  f"tumor/immune DEPTOR frac mean = "
                  f"{split['DEPTOR_tumor_frac'].mean():.3f}/{split['DEPTOR_immune_frac'].mean():.3f}")

    pd.DataFrame(beta_rows).to_csv(os.path.join(OUT, "empirical_compartment_beta.csv"), index=False)
    pd.DataFrame(inter_rows).to_csv(os.path.join(OUT, "empirical_interaction.csv"), index=False)
    print("\n=== empirical beta ===")
    print(pd.DataFrame(beta_rows).to_string(index=False))
    print("\n=== empirical interaction (DEPTOR-split x RT) ===")
    print(pd.DataFrame(inter_rows).to_string(index=False))
    print(f"\nDONE -> {OUT}")


if __name__ == "__main__":
    main()
