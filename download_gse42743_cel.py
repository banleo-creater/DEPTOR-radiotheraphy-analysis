#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Download the 103 GSE42743 CEL.gz files in parallel from NCBI GEO.

Single-stream throughput through the local proxy is ~30 KB/s, so the 583 MB
GSE42743_RAW.tar would take hours. Parallel per-sample downloads aggregate
bandwidth (same trick that worked for GSE181300).

Output: data/gse42743/cel/<GSM>_<title>.CEL.gz
"""
import os, re, sys, time, urllib.request, concurrent.futures, shutil

BASE = r"WorkBuddy/2026-10-17-11-15-46"
SM = os.path.join(BASE, "data", "gse42743", "GSE42743_series_matrix.txt")
OUT = os.path.join(BASE, "data", "gse42743", "cel")
os.makedirs(OUT, exist_ok=True)

MAX_WORKERS = 4


def parse_urls():
    with open(SM, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("!Sample_supplementary_file"):
                parts = [p.strip().strip('"') for p in line.rstrip("\n").split("\t")[1:]]
                urls = [p for p in parts if p.lower().endswith(".cel.gz")]
                return urls
    return []


def fetch(url):
    fname = url.rsplit("/", 1)[-1]
    dest = os.path.join(OUT, fname)
    if os.path.exists(dest) and os.path.getsize(dest) > 100_000:
        return (fname, "skip", os.path.getsize(dest))
    tmp = dest + ".part"
    last = ""
    for attempt in range(8):
        try:
            # ft<PATH>
            u = url.replace("ft<PATH>", "http<PATH>")
            req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=180) as r, open(tmp, "wb") as w:
                shutil.copyfileobj(r, w)
            if os.path.getsize(tmp) < 100_000:
                raise IOError(f"too small: {os.path.getsize(tmp)}")
            os.replace(tmp, dest)
            return (fname, "ok", os.path.getsize(dest))
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}"
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass
            # 503 = rate limited -> long backoff
            wait = 15 * (attempt + 1) if e.code == 503 else 3 * (attempt + 1)
            time.sleep(wait)
        except Exception as e:
            last = str(e)[:120]
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass
            time.sleep(3 + attempt * 3)
    return (fname, f"FAIL: {last}", 0)


def main():
    urls = parse_urls()
    print(f"CEL URLs: {len(urls)}")
    if not urls:
        raise SystemExit("no CEL urls")
    t0 = time.time()
    ok = skip = fail = 0
    total_bytes = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = [ex.submit(fetch, u) for u in urls]
        for i, fu in enumerate(concurrent.futures.as_completed(futs), 1):
            fn, st, sz = fu.result()
            total_bytes += sz
            if st == "ok":
                ok += 1
            elif st == "skip":
                skip += 1
            else:
                fail += 1
                print("  FAIL", fn, st, flush=True)
            if i % 10 == 0 or i == len(urls):
                el = time.time() - t0
                rate = total_bytes / 1024 / max(el, 1)
                print(f"  [{i}/{len(urls)}] ok={ok} skip={skip} fail={fail} "
                      f"| {total_bytes/1024/1024:.1f} MB | {rate:.0f} KB/s | {el:.0f}s",
                      flush=True)
    print(f"\nDONE. ok={ok} skip={skip} fail={fail} in {time.time()-t0:.0f}s")
    if fail:
        raise SystemExit(f"{fail} files failed")


if __name__ == "__main__":
    main()
