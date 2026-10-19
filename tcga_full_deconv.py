#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TCGA-HNSC 全转录组反卷积（修正 249 基因面板偏倚）
=================================================
用 Xena HiSeqV2 全基因矩阵（20531 基因）替代 cBioPortal 的 249 基因子集，
转 log2->linear 后喂给 EPIC 与 quanTIseq，得到与 GSE65858/GSE41613 一致的、
可信的肿瘤 vs 免疫 DEPTOR 分室估计。

严格复用 epic_quantiseq_deconv 主块的调用顺序:
    prop, efrac = run_epic(tpm)        # efrac = 细胞分数(含 otherCells)
    qfrac       = run_quantiseq(tpm)
    esplit      = split_depTOR(efrac, EPIC_MAP, DEPTOR_C)
    qsplit      = split_depTOR(qfrac, QS_MAP, DEPTOR_C)

输出:
  data/results_epic_quantiseq/TCGA_full_EPIC_DEPTOR_split.csv
  data/results_epic_quantiseq/TCGA_full_quantiseq_DEPTOR_split.csv
  data/results_epic_quantiseq/TCGA_full_EPIC_fractions.csv
  data/results_epic_quantiseq/TCGA_full_quantiseq_fractions.csv
  data/results_epic_quantiseq/TCGA_full_validation.csv  (249面板 vs 全转录组 交叉比对)
"""
import os, sys, gzip
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import epic_quantiseq_deconv as eq
from scipy.stats import pearsonr, spearmanr

DATA = os.path.join(ROOT, "data")
XENA = os.path.join(DATA, "tcga_hnsc_hiseq.gz")
CLIN = os.path.join(DATA, "tcga_hnsc_pca_deptor.csv")
OUT = os.path.join(DATA, "results_epic_quantiseq")
os.makedirs(OUT, exist_ok=True)


def load_xena_samples_x_genes():
    """返回 samples x genes 的线性(RSEM)矩阵，索引为 sampleId，列为 Gene Symbol。"""
    print("loading Xena HiSeqV2 (log2 -> linear) ...")
    genes, samples, rows = [], [], []
    with gzip.open(XENA, "rt") as f:
        header = f.readline().rstrip("\n").split("\t")
        samples = header[1:]
        for line in f:
            p = line.rstrip("\n").split("\t")
            genes.append(p[0])
            vals = np.array([float(v) for v in p[1:]], dtype=float)
            rows.append((2.0 ** vals) - 1.0)     # log2(RSEM+1) -> linear
    M = pd.DataFrame(np.array(rows), index=genes, columns=samples)  # genes x samples
    M = M[~M.index.duplicated(keep="first")]
    print(f"  raw matrix: {M.shape[0]} genes x {M.shape[1]} samples")
    return M.T                                      # samples x genes


def main():
    clin = pd.read_csv(CLIN)
    keep = set(clin["sampleId"].astype(str))
    M = load_xena_samples_x_genes()
    sub = M.loc[[s for s in M.index if s in keep]]   # samples x genes, 仅肿瘤样本
    print(f"  matched tumor samples: {sub.shape[0]} / {len(keep)}")

    # 走与主块一致的 to_linear_tpm（Xena 已是 TOIL 归一化，is_array=False 不重排）
    tpm = eq.to_linear_tpm(sub, is_array=False)
    print(f"  TPM: {tpm.shape}")

    print("\n===== EPIC (full transcriptome) =====")
    prop_frac, efrac = eq.run_epic(tpm)
    print("\n===== quanTIseq (full transcriptome) =====")
    qfrac = eq.run_quantiseq(tpm, is_tumor=True)

    esplit, ecomp = eq.split_depTOR(efrac, eq.EPIC_MAP, eq.DEPTOR_C)
    qsplit, qcomp = eq.split_depTOR(qfrac, eq.QS_MAP, eq.DEPTOR_C)

    print("\nEPIC  mean cell fractions:", {k: round(float(v), 3) for k, v in
          efrac.mean().to_dict().items()})
    print("qTseq mean cell fractions:", {k: round(float(v), 3) for k, v in
          qfrac.mean().to_dict().items()})
    print("EPIC  tumor/immune DEPTOR: %.3f / %.3f" % (
        esplit["DEPTOR_tumor_frac"].mean(), esplit["DEPTOR_immune_frac"].mean()))
    print("qTseq tumor/immune DEPTOR: %.3f / %.3f" % (
        qsplit["DEPTOR_tumor_frac"].mean(), qsplit["DEPTOR_immune_frac"].mean()))

    efrac.to_csv(os.path.join(OUT, "TCGA_full_EPIC_fractions.csv"))
    qfrac.to_csv(os.path.join(OUT, "TCGA_full_quantiseq_fractions.csv"))
    esplit.to_csv(os.path.join(OUT, "TCGA_full_EPIC_DEPTOR_split.csv"))
    qsplit.to_csv(os.path.join(OUT, "TCGA_full_quantiseq_DEPTOR_split.csv"))

    # 与 249 基因面板的旧结果比对（验证全转录组是否消除 EPIC/quanTIseq 矛盾）
    val_rows = []
    for tag, old_fn, new_split in [("EPIC", "TCGA_EPIC_DEPTOR_split.csv", esplit),
                                   ("qTseq", "TCGA_quantiseq_DEPTOR_split.csv", qsplit)]:
        old = pd.read_csv(os.path.join(OUT, old_fn), index_col=0)
        cm = old.join(new_split.add_suffix("_full"), how="inner")
        for col, oc, nc in [("tumor", "DEPTOR_tumor_frac", "DEPTOR_tumor_frac_full"),
                            ("immune", "DEPTOR_immune_frac", "DEPTOR_immune_frac_full")]:
            if oc in cm and nc in cm:
                r = pearsonr(cm[oc].fillna(cm[oc].mean()),
                             cm[nc].fillna(cm[nc].mean()))[0]
                val_rows.append(dict(comparison=f"TCGA {tag} {col}: 249panel_vs_full",
                                     r=round(float(r), 3)))
    # 全转录组下 EPIC vs quanTIseq 一致性（应转为正向、与 GSE 队列一致）
    cm2 = esplit.join(qsplit.add_suffix("_q"), how="inner")
    for col, ec, qc in [("tumor", "DEPTOR_tumor_frac", "DEPTOR_tumor_frac_q"),
                        ("immune", "DEPTOR_immune_frac", "DEPTOR_immune_frac_q")]:
        if ec in cm2 and qc in cm2:
            r = pearsonr(cm2[ec].fillna(cm2[ec].mean()),
                         cm2[qc].fillna(cm2[qc].mean()))[0]
            val_rows.append(dict(comparison=f"TCGA full {col}: EPIC_vs_qTseq",
                                 r=round(float(r), 3)))
    pd.DataFrame(val_rows).to_csv(
        os.path.join(OUT, "TCGA_full_validation.csv"), index=False)
    print("\n=== validation ===")
    print(pd.DataFrame(val_rows).to_string(index=False))
    print(f"\nDONE -> {OUT}")


if __name__ == "__main__":
    main()
