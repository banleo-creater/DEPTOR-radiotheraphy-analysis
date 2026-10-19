#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-cohort PSEUDOBULK replication of the
   "myeloid DEPTOR  <->  reduced NK abundance"
association found in GSE164690 (Kürten 2021).

Cohorts (all local, all HNSCC / OSCC scRNA-seq):
  1. GSE164690  CD45+ immune, 18 patients        (discovery cohort)
  2. GSE139324  CD45+ TIL,  ~26 patients, HPV annotated
                 (Immune landscape of viral- vs carcinogen-driven HNSCC)
  3. GSE198315  OSCC multiregion, 10 patients, tumour regions TC+TP
                 (268k cells; loaded from a pre-extracted gene subset)

Design
------
Per SAMPLE (patient):
  - de novo cell typing with ONE shared marker panel -> NK / Myeloid / T / B /
    Tumor / Stroma / Endo  (identical across cohorts)
  - myeloid pseudobulk: sum raw DEPTOR counts over myeloid cells, divided by the
    summed per-cell library size of those myeloid cells -> CPM -> log2(CPM+1).
    (Standard pseudobulk; far more robust than averaging log expression when a
    gene is sparse, which DEPTOR is.)
  - NK fraction = n_NK / n_immune   (immune = T + NK + Myeloid + B)
Then, per cohort: Spearman(myeloid DEPTOR pseudobulk, NK fraction).
Cross-cohort: (a) cohort-normalised ranks pooled, (b) Stouffer combination of
Fisher-z, (c) forest plot. Also HPV-stratified within GSE139324.

Why cohort-normalised ranks: absolute CPM is not comparable across platforms
(10x v2/v3, different chemistries/depths), so within-cohort rank/percentile is
used before pooling.
"""
import os, re, glob, gzip, warnings
import numpy as np, pandas as pd
import scipy.io, scipy.sparse as sp
from scipy.stats import spearmanr, pearsonr, norm as normdist
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

BASE = "WorkBuddy/2026-10-17-11-15-46"
NDRV = "<PATH>"
OUT  = os.path.join(BASE, "data", "results_multicohort"); os.makedirs(OUT, exist_ok=True)

# ---------------- shared panel ----------------
MARKERS = {
    "NK":      ["NCAM1","NKG7","KLRF1","KLRD1","GNLY","TBX21","PRF1","GZMB","KLRK1","XCL1","XCL2","FCGR3A"],
    "Myeloid": ["LYZ","CD68","CD163","CSF1R","FCGR1A","ITGAM","VSIG4","MRC1","HLA-DRA","CD14","SPI1","AIF1","LST1"],
    "Tcell":   ["CD3D","CD3E","CD8A","CD8B","CD4","IL7R","CD2"],
    "Bcell":   ["CD79A","CD79B","MS4A1","JCHAIN","MZB1","CD19"],
    "Tumor":   ["EPCAM","KRT5","KRT14","KRT8","KRT18","TP63","KRT19","SFN","ANXA1"],
    "Stroma":  ["COL1A1","COL1A2","ACTA2","PDGFRA","PDGFRB","DCN","LUM","FAP","MMP2"],
    "Endo":    ["PECAM1","CLDN5","VWF","CDH5"],
}
TYPES = list(MARKERS.keys())
IMMUNE = {"NK", "Myeloid", "Tcell", "Bcell"}
TARGET = sorted({g for v in MARKERS.values() for g in v} | {"DEPTOR", "PTPRC"})

MIN_TOT   = 200     # per-cell min total counts
MIN_MYE   = 20      # min myeloid cells per sample to be usable
MIN_IMM   = 100     # min immune cells per sample

# ---------------- io helpers ----------------
def read_10x(bc_p, ft_p, mx_p):
    with gzip.open(bc_p, "rt") as f:
        barcodes = [l.strip() for l in f if l.strip()]
    syms = []
    with gzip.open(ft_p, "rt") as f:
        for l in f:
            p = l.rstrip("\n").split("\t")
            syms.append(p[1] if len(p) > 1 else p[0])
    M = scipy.io.mmread(mx_p).tocsc()          # genes x cells
    return barcodes, syms, M

def subset_sample(barcodes, syms, M, min_tot=MIN_TOT):
    """-> (X_target cells x n_target (raw), totals, gene_names)"""
    totals = np.asarray(M.sum(axis=0)).ravel()
    idx = [i for i, s in enumerate(syms) if s in set(TARGET)]
    X = M[idx, :].T.tocsr()                    # cells x target (raw counts)
    names = [syms[i] for i in idx]
    keep = totals >= min_tot
    return X[keep].tocsc(), totals[keep], names, np.array(barcodes)[keep]

# ---------------- cohort loaders ----------------
def load_gse164690():
    """CD45+ immune, patients HN01..HN18."""
    d = os.path.join(NDRV, "GSE164690", "raw_extracted")
    out = []
    for bc in sorted(glob.glob(os.path.join(d, "*_CD45p_barcodes.tsv.gz"))):
        base = bc.replace("_barcodes.tsv.gz", "")
        m = re.search(r"(HN\d+)_", os.path.basename(base))
        if not m: continue
        try:
            b, s, M = read_10x(bc, base + "_features.tsv.gz", base + "_matrix.mtx.gz")
            X, tot, names, bcodes = subset_sample(b, s, M)
            out.append(dict(cohort="GSE164690", patient=m.group(1), X=X, totals=tot,
                            genes=names, hpv="NA", tissue="tumour immune"))
        except Exception as e:
            print("   skip", os.path.basename(base), e)
    return out

def load_gse139324():
    """TIL (tumour-infiltrating CD45+), patients HNSCC_n, HPV annotated."""
    d = os.path.join(NDRV, "GSE139324", "raw_extracted")
    sm = os.path.join(NDRV, "GSE139324", "GSE139324_series_matrix.txt.gz")
    hpv_map = {}
    try:
        with gzip.open(sm, "rt", errors="ignore") as f:
            titles = None
            for line in f:
                if line.startswith("!Sample_title"):
                    titles = [v.strip('"') for v in line.rstrip("\n").split("\t")[1:]]
                elif line.startswith("!Sample_characteristics_ch1"):
                    vals = [v.strip('"') for v in line.rstrip("\n").split("\t")[1:]]
                    if vals and any("hpv_status" in v for v in vals):
                        hp = [v.split(":")[-1].strip() for v in vals]
                        if titles: hpv_map = dict(zip(titles, hp))
    except Exception as e:
        print("   series_matrix parse failed:", e)
    out = []
    for bc in sorted(glob.glob(os.path.join(d, "*_TIL_barcodes.tsv.gz"))):
        base = bc.replace("_barcodes.tsv.gz", "")
        title = os.path.basename(base).split("_", 1)[-1]      # HNSCC_1_TIL
        m = re.search(r"(HNSCC_\d+)_TIL", title)
        if not m: continue
        ft = base + "_genes.tsv.gz"
        if not os.path.exists(ft): ft = base + "_features.tsv.gz"
        try:
            b, s, M = read_10x(bc, ft, base + "_matrix.mtx.gz")
            X, tot, names, bcodes = subset_sample(b, s, M)
            out.append(dict(cohort="GSE139324", patient=m.group(1), X=X, totals=tot,
                            genes=names, hpv=hpv_map.get(title, "NA"), tissue="TIL"))
        except Exception as e:
            print("   skip", title, e)
    return out

def load_gse198315():
    """OSCC multiregion; kept regions TC (core) + TP (periphery), aggregated per patient."""
    src = os.path.join(NDRV, "GSE198315")
    prep = os.path.join(BASE, "data", "gse198315_prep")
    sub = os.path.join(prep, "subset_raw.txt")
    idx_f = os.path.join(prep, "target_idx.txt")
    if not os.path.exists(sub):
        print("   GSE198315 subset not ready - skipped")
        return []
    # barcodes FIRST: they define the true number of cells (268,131)
    with gzip.open(os.path.join(src, "GSE198315_barcodes.tsv.gz"), "rt") as f:
        bcs = [l.strip() for l in f if l.strip()]
    ncells = len(bcs)
    with open(idx_f) as f:
        tidx = [int(l.strip()) for l in f if l.strip()]
    with gzip.open(os.path.join(src, "GSE198315_features.tsv.gz"), "rt") as f:
        syms = [l.rstrip("\n").split("\t")[1] for l in f]
    gnames = [syms[i - 1] for i in tidx]                 # names in target_idx order
    # read subset (rows: gene_idx \t cell_idx \t value ; TOT rows carry totals)
    gi, ci, vv, tot_map = [], [], [], {}
    with open(sub) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) != 3: continue
            if p[0] == "TOT":
                tot_map[int(p[1])] = float(p[2]); continue
            gi.append(int(p[0])); ci.append(int(p[1])); vv.append(float(p[2]))
    if not tot_map:
        print("   GSE198315 subset incomplete (no TOT rows) - skipped")
        return []
    totals = np.array([tot_map.get(i + 1, 0.0) for i in range(ncells)])
    # map gene_idx -> column order in TARGET
    idx2col = {g_idx: k for k, g_idx in enumerate(tidx)}
    rows = np.array([idx2col[g] for g in gi]); cols = np.array(ci) - 1
    X = sp.coo_matrix((np.array(vv), (cols, rows)), shape=(ncells, len(tidx))).tocsr()
    prefix = np.array([re.sub(r"_.+$", "", b) for b in bcs])   # e.g. P01-TC
    out = []
    for pref in sorted(set(prefix)):
        pat_m, les_m = re.match(r"(P\d+)-(\w+)", pref).groups()
        mask = prefix == pref
        Xs, ts = X[mask], totals[mask]
        keep = ts >= MIN_TOT
        if keep.sum() < MIN_IMM: continue
        out.append(dict(cohort="GSE198315", patient=pat_m, lesion=les_m,
                        X=Xs[keep], totals=ts[keep], genes=gnames,
                        hpv="NA", tissue=f"OSCC-{les_m}"))
    return out

# ---------------- per-sample processing ----------------
def process(rec):
    X = rec["X"].tocsc(); tot = np.asarray(rec["totals"]).ravel()
    genes = rec["genes"]
    gmap = {}
    for k, g in enumerate(genes):
        gmap.setdefault(g, k)
    # normalise: counts -> CP10K -> log1p
    scale = 1e4 / np.maximum(tot, 1.0)
    Xn = sp.diags(scale) @ X
    Xn = Xn.tocsr(); Xn.data = np.log1p(Xn.data)
    def sc(mk):
        idxs = [gmap[g] for g in mk if g in gmap]
        if not idxs: return np.zeros(Xn.shape[0])
        return np.asarray(Xn[:, idxs].mean(axis=1)).ravel()
    S = np.column_stack([sc(MARKERS[t]) for t in TYPES])
    lab = np.argmax(S, axis=1)
    lab[S.max(axis=1) < 0.05] = -1
    names = np.array(TYPES + ["Unassigned"])
    labels = np.array([names[i] for i in lab], dtype=object)
    imm = np.isin(labels, list(IMMUNE))
    nk = int((labels == "NK").sum()); mye = labels == "Myeloid"
    n_imm = int(imm.sum()); n_mye = int(mye.sum())
    if n_imm < MIN_IMM or n_mye < MIN_MYE:
        return None
    # myeloid pseudobulk DEPTOR: sum(raw counts) / sum(library) -> CPM -> log2
    if "DEPTOR" in gmap:
        dc = np.asarray(X[:, gmap["DEPTOR"]].todense()).ravel()
        dsum = float(dc[mye].sum()); lib = float(tot[mye].sum())
        cpm = dsum / lib * 1e6 if lib > 0 else np.nan
        pb = float(np.log2(cpm + 1))
        # sensitivity: mean log1p normalised expression across myeloid cells
        dn = np.asarray(Xn[:, gmap["DEPTOR"]].todense()).ravel()
        mean_log = float(dn[mye].mean()); detect = float((dc[mye] > 0).mean())
    else:
        pb = cpm = mean_log = detect = np.nan
    return dict(cohort=rec["cohort"], patient=rec["patient"],
                lesion=rec.get("lesion", "NA"), hpv=rec.get("hpv", "NA"),
                n_cells=int(len(labels)), n_immune=n_imm, n_NK=nk, n_myeloid=n_mye,
                NK_fraction=nk / n_imm if n_imm else np.nan,
                DEPTOR_pseudobulk_log2CPM=pb, DEPTOR_myeloid_CPM=cpm,
                DEPTOR_myeloid_meanlog=mean_log, DEPTOR_myeloid_detect=detect)

# ---------------- run ----------------
print("Loading cohorts ...")
samples = []
for fn, nm in [(load_gse164690, "GSE164690"), (load_gse139324, "GSE139324"),
               (load_gse198315, "GSE198315")]:
    recs = fn()
    print(f"  {nm}: {len(recs)} raw units loaded")
    ok = 0
    for r in recs:
        p = process(r)
        if p: samples.append(p); ok += 1
    print(f"  {nm}: {ok} usable samples")

df = pd.DataFrame(samples)
df.to_csv(os.path.join(OUT, "multicohort_samples.csv"), index=False)
print("\n=== usable samples per cohort ===")
print(df.groupby("cohort").agg(n=("patient", "size"),
      n_NK=("n_NK", "sum"), NK_frac=("NK_fraction", "mean")).to_string())

# ---------------- per-cohort correlations ----------------
print("\n=== Per-cohort Spearman: myeloid DEPTOR pseudobulk vs NK fraction ===")
rows = []
for c, sub in df.groupby("cohort"):
    s = sub[["DEPTOR_pseudobulk_log2CPM", "NK_fraction"]].dropna()
    if len(s) < 5:
        print(f"  {c}: n={len(s)} too few"); continue
    rho, p = spearmanr(s.iloc[:, 0], s.iloc[:, 1])
    r, pr = pearsonr(s.iloc[:, 0], s.iloc[:, 1])
    rows.append(dict(cohort=c, n=len(s), Spearman_rho=round(rho, 3), Spearman_p=round(p, 4),
                     Pearson_r=round(r, 3), Pearson_p=round(pr, 4)))
    print(f"  {c:11s} n={len(s):3d}  rho={rho:+.3f} (p={p:.3g})  r={r:+.3f} (p={pr:.3g})")
coh = pd.DataFrame(rows)

# sensitivity with mean-log DEPTOR
print("\n=== Sensitivity: mean log1p DEPTOR (instead of pseudobulk) ===")
for c, sub in df.groupby("cohort"):
    s = sub[["DEPTOR_myeloid_meanlog", "NK_fraction"]].dropna()
    if len(s) >= 5:
        rho, p = spearmanr(s.iloc[:, 0], s.iloc[:, 1])
        print(f"  {c:11s} n={len(s):3d}  rho={rho:+.3f} (p={p:.3g})")

# ---------------- cross-cohort ----------------
print("\n=== Cross-cohort combination ===")
d2 = df.dropna(subset=["DEPTOR_pseudobulk_log2CPM", "NK_fraction"]).copy()
# within-cohort percentile so absolute CPM is comparable across platforms
d2["DEPTOR_pct"] = d2.groupby("cohort")["DEPTOR_pseudobulk_log2CPM"].rank(pct=True)
rho_p, p_p = spearmanr(d2["DEPTOR_pct"], d2["NK_fraction"])
print(f"  Pooled (cohort-normalised ranks, n={len(d2)}): rho={rho_p:+.3f} (p={p_p:.3g})")
# Stouffer combination of Fisher z
zs, ns = [], []
for _, r_ in coh.iterrows():
    n = r_["n"]; rho = r_["Spearman_rho"]
    if n > 3 and abs(rho) < 0.999:
        z = 0.5 * np.log((1 + rho) / (1 - rho)) * np.sqrt(n - 3)
        zs.append(z); ns.append(n)
if zs:
    Z = np.sum(zs) / np.sqrt(len(zs))
    pZ = 2 * (1 - normdist.cdf(abs(Z)))
    print(f"  Stouffer combined Z={Z:+.2f} (p={pZ:.3g}) across {len(zs)} cohorts")
    print(f"    per-cohort z: {[round(z,2) for z in zs]}")

# ---------------- HPV stratification (GSE139324 has hpv_status) ----------------
print("\n=== HPV stratification (GSE139324) ===")
g = df[df.cohort == "GSE139324"]
for h in sorted(g.hpv.dropna().unique()):
    s = g[g.hpv == h][["DEPTOR_pseudobulk_log2CPM", "NK_fraction"]].dropna()
    if len(s) >= 5:
        rho, p = spearmanr(s.iloc[:, 0], s.iloc[:, 1])
        print(f"  {h:14s} n={len(s):3d}  rho={rho:+.3f} (p={p:.3g})")

coh.to_csv(os.path.join(OUT, "multicohort_correlations.csv"), index=False)

# ---------------- figures ----------------
fig, ax = plt.subplots(figsize=(7, 4))
if len(coh):
    ypos = np.arange(len(coh))
    for i, r_ in coh.iterrows():
        n, rho = r_["n"], r_["Spearman_rho"]
        se = 1 / np.sqrt(max(n - 3, 1))
        lo, hi = np.tanh(np.arctanh(np.clip(rho, -.999, .999)) - 1.96 * se), \
                 np.tanh(np.arctanh(np.clip(rho, -.999, .999)) + 1.96 * se)
        ax.errorbar(rho, i, xerr=[[rho - lo], [hi - rho]], fmt="o", color="#1f77b4", capsize=4)
        ax.text(hi + 0.03, i, f"n={n}, p={r_['Spearman_p']:.3f}", va="center", fontsize=7)
ax.axvline(0, color="k", lw=.8)
ax.set_yticks(range(len(coh))); ax.set_yticklabels(coh["cohort"], fontsize=9)
ax.set_xlabel("Spearman rho (myeloid DEPTOR pseudobulk vs NK fraction)")
ax.set_title("Multi-cohort replication\n(negative = higher myeloid DEPTOR -> fewer NK)")
ax.set_xlim(-1, 1); plt.tight_layout()
plt.savefig(os.path.join(OUT, "multicohort_forest.png"), dpi=140); plt.close()

# scatter per cohort
fig, axes = plt.subplots(1, max(len(df.cohort.unique()), 1), figsize=(5 * len(df.cohort.unique()), 4.5))
if len(df.cohort.unique()) == 1: axes = [axes]
for ax, c in zip(np.atleast_1d(axes), sorted(df.cohort.unique())):
    s = df[df.cohort == c]
    ax.scatter(s["DEPTOR_pseudobulk_log2CPM"], s["NK_fraction"], s=45, edgecolor="k", alpha=.8)
    ss = s[["DEPTOR_pseudobulk_log2CPM", "NK_fraction"]].dropna()
    if len(ss) >= 5:
        rho, p = spearmanr(ss.iloc[:, 0], ss.iloc[:, 1])
        ax.set_title(f"{c}\nrho={rho:+.2f} (p={p:.3g}), n={len(ss)}", fontsize=9)
    ax.set_xlabel("myeloid DEPTOR pseudobulk (log2 CPM+1)"); ax.set_ylabel("NK / immune")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "multicohort_scatter_bycohort.png"), dpi=140); plt.close()

# pooled scatter (cohort-normalised ranks)
fig, ax = plt.subplots(figsize=(5.5, 4.5))
for c in sorted(d2.cohort.unique()):
    s = d2[d2.cohort == c]
    ax.scatter(s["DEPTOR_pct"], s["NK_fraction"], s=42, alpha=.8, edgecolor="k", label=c)
ax.set_xlabel("myeloid DEPTOR (within-cohort percentile)"); ax.set_ylabel("NK / immune")
ax.set_title(f"Pooled (n={len(d2)}): rho={rho_p:+.2f} (p={p_p:.3g})")
ax.legend(fontsize=7); plt.tight_layout()
plt.savefig(os.path.join(OUT, "multicohort_pooled.png"), dpi=140); plt.close()

print("\nSaved to", OUT, "DONE.")
