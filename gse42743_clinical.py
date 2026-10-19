#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Parse GSE42743 clinical metadata from the GEO series matrix.

GSE42743 (Lohavanichbutr et al., oral cavity OSCC, GPL570, n=~120) stores its
per-sample characteristics in a *staggered* layout: the same
`!Sample_characteristics_ch1` row holds different keys for different samples.
So we must collect ALL characteristic strings per sample column, then split
each on the first ":".

Outputs
  data/gse42743_clinical.csv  (index = GSM)
     rt        : 1 if any radiation (R / RT / cRT), else 0
     treat_raw : original treatment string
     os_t      : overall-survival time (months)  = futime/30.4375
     os_e      : overall death (alldeath)
     dss_t     : disease-specific time (months)  = futime/30.4375
     dss_e     : OSCC death (osccdeath)
     ajcc4, ajcc2, t_stage, cn_stage, pn_stage, ecs, age, sex, site, hpv_note
"""
import os, re, csv
import pandas as pd

BASE = r"WorkBuddy/2026-10-17-11-15-46"
SM   = os.path.join(BASE, "data", "gse42743", "GSE42743_series_matrix.txt")
OUT  = os.path.join(BASE, "data", "gse42743_clinical.csv")


def split_quoted(line):
    """Split a GEO matrix line on tabs, stripping surrounding quotes."""
    parts = line.rstrip("\n").split("\t")
    if parts:
        parts[0] = parts[0].lstrip("!")
    return [p.strip().strip('"') for p in parts]


def main():
    gsm, title = None, None
    chars = []          # list of lists (one per characteristics row)
    with open(SM, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("!Sample_geo_accession"):
                gsm = split_quoted(line)[1:]
            elif line.startswith("!Sample_title"):
                title = split_quoted(line)[1:]
            elif line.startswith("!Sample_characteristics_ch1"):
                chars.append(split_quoted(line)[1:])
            elif line.startswith("!series_matrix_table_begin"):
                break

    if gsm is None:
        raise SystemExit("no !Sample_geo_accession found")
    n = len(gsm)

    rows = []
    for i in range(n):
        rec = {"gsm": gsm[i], "title": title[i] if title and i < len(title) else ""}
        # collect all key:value strings for this sample
        for crow in chars:
            if i >= len(crow):
                continue
            v = crow[i].strip()
            if not v or ":" not in v:
                continue
            k, val = v.split(":", 1)
            k = k.strip().lower()
            val = val.strip()
            # keep first occurrence (later blank/duplicate slots must not overwrite)
            if k not in rec or rec[k] == "":
                rec[k] = val
        rows.append(rec)

    df = pd.DataFrame(rows)

    def num(x):
        try:
            return float(str(x).strip())
        except Exception:
            return None

    raw = df.to_dict("records")
    out = []
    for r in raw:
        treat = str(r.get("treatment*", r.get("treatment", "")) or "")
        tl = treat.lower()

        # ---- radiation present?  (R, RT, cRT, "radiation")
        has_rt = int(bool(re.search(r"\brt\b|\br\b|crt|radiat", tl)))
        # surgery-only / induction-only = no RT
        # note: "s" alone = surgery; "ic" = induction chemo (no radiation)
        futime = num(r.get("futime", ""))
        os_t = futime / 30.4375 if futime is not None else None
        osccdeath = r.get("osccdeath", "")
        alldeath  = r.get("alldeath", "")
        vital     = str(r.get("vital", ""))

        # fallbacks when osccdeath / alldeath blank
        if alldeath in ("", None):
            alldeath = 1 if vital.lower().startswith("dead") else (0 if vital.lower().startswith("alive") else None)
        if osccdeath in ("", None):
            if "oscc" in vital.lower() or "oral ca" in vital.lower() or "with disease" in vital.lower():
                osccdeath = 1
            elif vital.lower().startswith("alive") or "ned" in vital.lower():
                osccdeath = 0
            else:
                osccdeath = None          # Dead-unknown / Dead-other -> not disease-specific
        try:
            os_e = int(float(alldeath)) if alldeath not in (None, "") else None
        except Exception:
            os_e = None
        try:
            dss_e = int(float(osccdeath)) if osccdeath not in (None, "") else None
        except Exception:
            dss_e = None

        out.append({
            "gsm": r["gsm"],
            "title": r["title"],
            "rt": has_rt,
            "treat_raw": treat,
            "os_t": os_t,
            "os_e": os_e,
            "dss_t": os_t,
            "dss_e": dss_e,
            "ajcc4": r.get("ajcc4", ""),
            "ajcc2": r.get("ajcc2", ""),
            "t_stage": r.get("t stage", ""),
            "cn_stage": r.get("cn stage", ""),
            "pn_stage": r.get("pn stage", ""),
            "ecs": r.get("ecs", ""),
            "age": r.get("age@dx", ""),
            "sex": r.get("gender", ""),
            "site": r.get("oc vs. op", ""),
            "smoking": r.get("smoking status", ""),
            "vital": vital,
        })

    o = pd.DataFrame(out).set_index("gsm")
    o.to_csv(OUT)

    # ---- diagnostics
    print(f"Parsed {len(o)} samples -> {OUT}\n")
    print("=== treatment 原始取值分布 ===")
    print(o["treat_raw"].value_counts(dropna=False).to_string())
    print("\n=== RT 分组 ===")
    print(o["rt"].value_counts(dropna=False).to_string())
    print("\n=== OS 事件 ===")
    print(o["os_e"].value_counts(dropna=False).to_string())
    print("\n=== DSS(OSCC 死亡) 事件 ===")
    print(o["dss_e"].value_counts(dropna=False).to_string())
    print("\n=== avialability ===")
    print("os_t non-null:", o["os_t"].notna().sum(),
          "| os_e non-null:", o["os_e"].notna().sum(),
          "| dss_e non-null:", o["dss_e"].notna().sum())
    print("\n=== RT × OS event 交叉 ===")
    print(pd.crosstab(o["rt"], o["os_e"], dropna=False).to_string())
    print("\n=== RT × DSS event 交叉 ===")
    print(pd.crosstab(o["rt"], o["dss_e"], dropna=False).to_string())
    print("\n=== 前 8 行 ===")
    print(o.head(8).to_string())


if __name__ == "__main__":
    main()
