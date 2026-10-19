#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Download GSE181300 Visium supplementary files (selected, analysis-essential only)
from NCBI GEO. Pulls, per sample:
  - filtered_feature_bc_matrix.h5
  - scalefactors_json.json.gz
  - tissue_positions_list.csv.gz
  - tissue_hires_image.png.gz   (for histology overlay)
and arranges them in 10x Visium layout:
  <OUT>/<sample>/filtered_feature_bc_matrix.h5
  <OUT>/<sample>/spatial/{scalefactors_json.json, tissue_positions_list.csv, tissue_hires_image.png}

Uses ThreadPoolExecutor for parallel HTTPS downloads (bypasses single-stream throttle).
"""
import os, gzip, shutil, urllib.request, concurrent.futures, time

BASE = "WorkBuddy/2026-10-17-11-15-46"
OUT  = os.path.join(BASE, "data", "gse181300_raw")
os.makedirs(OUT, exist_ok=True)

# (gsm, sample_name, patient, region)
SAMPLES = [
    ("GSM5494475","HNSCC201125T04","upn1","invasive_front"),
    ("GSM5494476","HNSCC201125T05","upn1","invasive_front"),
    ("GSM5494477","HNSCC201125T07","upn1","inner_site"),
    ("GSM5494478","HNSCC201125T10","upn1","inner_site"),
    ("GSM5494479","P210325T1","upn2","invasive_front"),
    ("GSM5494480","P210325T3","upn2","invasive_front"),
    ("GSM5494481","P210325T5","upn2","inner_site"),
    ("GSM5494482","P210325T6","upn2","inner_site"),
]

# (suffix, target_subpath, decompress?)
FILES = [
    ("_filtered_feature_bc_matrix.h5",            "{name}/filtered_feature_bc_matrix.h5", False),
    ("_scalefactors_json.json.gz",                "{name}/spatial/scalefactors_json.json", True),
    ("_tissue_positions_list.csv.gz",            "{name}/spatial/tissue_positions_list.csv", True),
    ("_tissue_hires_image.png.gz",               "{name}/spatial/tissue_hires_image.png", True),
]

def build_jobs():
    jobs = []
    for gsm, name, pat, region in SAMPLES:
        for suf, tpl, dec in FILES:
            fname = f"{gsm}_{name}{suf}"
            url = f"http<PATH>"
            dest = os.path.join(OUT, tpl.format(name=name))
            jobs.append((url, dest, dec, fname))
    return jobs

def fetch(job):
    url, dest, dec, fname = job
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return (fname, "skip-existing")
    tmp = dest + ".part"
    last = ""
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as w:
                shutil.copyfileobj(r, w)
            if dec:
                with gzip.open(tmp, "rb") as gz, open(dest, "wb") as w:
                    shutil.copyfileobj(gz, w)
                os.remove(tmp)
            else:
                os.replace(tmp, dest)
            return (fname, "ok")
        except Exception as e:
            last = str(e)
            if os.path.exists(tmp):
                try: os.remove(tmp)
                except: pass
            time.sleep(2 + attempt*2)
    return (fname, f"FAIL: {last}")

def main():
    jobs = build_jobs()
    print(f"Total download jobs: {len(jobs)}")
    ok=0; fail=0; skip=0
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(fetch, j) for j in jobs]
        for fu in concurrent.futures.as_completed(futs):
            fn, st = fu.result()
            if st=="ok": ok+=1
            elif st=="skip-existing": skip+=1
            else:
                fail+=1; print("  FAIL", fn, st)
            print(f"  [{ok+skip+fail}/{len(jobs)}] {fn}: {st}")
    print(f"\nDONE. ok={ok} skip={skip} fail={fail}")
    if fail: raise SystemExit(f"{fail} files failed")

if __name__ == "__main__":
    main()
