#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Download GSE300147 (Xenium, HNSCC) — analysis-essential files only.

Per sample we need just TWO files (Xenium uploads also include huge
transcripts.parquet / morphology.ome.tif / cell_boundaries, which we skip):
  - *_cell_feature_matrix.h5   (cells x 476 genes)
  - *_cells.parquet.gz          (cell centroid coordinates + QC)

Excludes the ameloblastoma sample (GSM9054488, "Patient 10").
"""
import os, re, gzip, shutil, urllib.request, concurrent.futures, time

BASE = "WorkBuddy/2026-10-17-11-15-46"
RAW  = os.path.join(BASE, "data", "gse300147", "raw")
os.makedirs(RAW, exist_ok=True)

SERIES = os.path.join(BASE, "data", "gse300147", "series_matrix.txt.gz")
EXCLUDE = {"GSM9054488"}   # ameloblastoma, not HNSCC

def parse_urls():
    """Return {gsm: {'h5': url, 'cells': url}} from the series matrix."""
    import gzip as gz
    out = {}
    with gz.open(SERIES, "rt", errors="ignore") as f:
        for line in f:
            if not line.startswith("!Sample_supplementary_file"):
                continue
            for u in re.findall(r'ft<PATH>"]+', line):
                # filename (NOT the path) carries the true accession:
                #   .../GSM9054nnn/GSM9054471/suppl/GSM9054471_Nov_21_24_..._cells.parquet.gz
                fn = u.rsplit("/", 1)[-1]
                gsm_m = re.match(r"(GSM\d+)_", fn)
                if not gsm_m:
                    continue
                gsm = gsm_m.group(1)
                if gsm in EXCLUDE:
                    continue
                d = out.setdefault(gsm, {})
                if fn.endswith("_cell_feature_matrix.h5"):
                    d["h5"] = u.replace("ft<PATH>", "http<PATH>")
                elif fn.endswith("_cells.parquet.gz"):
                    d["cells"] = u.replace("ft<PATH>", "http<PATH>")
    return out

def fetch(job):
    url, dest, dec, label = job
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return (label, "skip")
    tmp = dest + ".part"
    last = ""
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=180) as r, open(tmp, "wb") as w:
                shutil.copyfileobj(r, w)
            if dec:
                with gzip.open(tmp, "rb") as gz, open(dest, "wb") as w:
                    shutil.copyfileobj(gz, w)
                os.remove(tmp)
            else:
                os.replace(tmp, dest)
            return (label, "ok")
        except Exception as e:
            last = str(e)
            if os.path.exists(tmp):
                try: os.remove(tmp)
                except Exception: pass
            time.sleep(2 + attempt * 2)
    return (label, f"FAIL {last}")

def main():
    meta = parse_urls()
    jobs = []
    for gsm, d in sorted(meta.items()):
        if "h5" in d:
            jobs.append((d["h5"], os.path.join(RAW, gsm, "cell_feature_matrix.h5"), False, f"{gsm}_h5"))
        if "cells" in d:
            jobs.append((d["cells"], os.path.join(RAW, gsm, "cells.parquet"), True, f"{gsm}_cells"))
    print(f"samples with h5: {sum(1 for d in meta.values() if 'h5' in d)}")
    print(f"samples with cells: {sum(1 for d in meta.values() if 'cells' in d)}")
    print(f"total jobs: {len(jobs)}")
    ok = skip = fail = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(fetch, j) for j in jobs]
        for fu in concurrent.futures.as_completed(futs):
            lab, st = fu.result()
            if st == "ok": ok += 1
            elif st == "skip": skip += 1
            else:
                fail += 1; print("  FAIL", lab, st)
    print(f"\nDONE ok={ok} skip={skip} fail={fail}")
    if fail:
        raise SystemExit(f"{fail} failed")

if __name__ == "__main__":
    main()
