#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Download GSE42743_RAW.tar with parallel HTTP Range requests.

The local proxy throttles every single stream to ~32 KB/s; the whole 556 MB tar
would take ~5 h. Parallel range requests on the SAME series-level URL aggregate
bandwidth (4 chunks gave ~124 KB/s, so 16 chunks should give ~0.5 MB/s).

Chunks are written to <dest>.part<idx>, then concatenated in order.
"""
import os, sys, time, subprocess, concurrent.futures

URL = "http<PATH>"
OUT = r"WorkBuddy/2026-10-17-11-15-46/data/gse42743/GSE42743_RAW.tar"
N = 16
CHUNK = 8 * 1024 * 1024     # 8 MB per range request; re-issued until complete


def content_length():
    out = subprocess.run(
        ["curl", "-sIL", "--ssl-no-revoke", "--max-time", "60", URL],
        capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.lower().startswith("content-length:"):
            return int(line.split(":")[1].strip())
    return None


def get_range(start, end, dest):
    for attempt in range(6):
        cmd = ["curl", "-sL", "--ssl-no-revoke",
               "--max-time", "300", "-r", f"{start}-{end}", "-o", dest, URL]
        subprocess.run(cmd, capture_output=True)
        if os.path.exists(dest) and os.path.getsize(dest) == (end - start + 1):
            return True
        time.sleep(2 + attempt * 3)
    return False


def main():
    total = content_length()
    if not total:
        raise SystemExit("cannot get content-length")
    print(f"total bytes = {total} ({total/1024/1024:.0f} MB)")

    parts = []
    start = 0
    while start < total:
        end = min(start + CHUNK - 1, total - 1)
        parts.append((start, end))
        start = end + 1
    print(f"chunks = {len(parts)}")

    # split chunks across N workers (each worker does a stride of chunks)
    def worker(idx):
        done = 0
        for ci in range(idx, len(parts), N):
            s, e = parts[ci]
            d = f"{OUT}.part{ci:04d}"
            if os.path.exists(d) and os.path.getsize(d) == (e - s + 1):
                done += e - s + 1
                continue
            if get_range(s, e, d):
                done += e - s + 1
            else:
                print(f"  chunk {ci} FAILED", flush=True)
        return done

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=N) as ex:
        futs = [ex.submit(worker, i) for i in range(N)]
        last = 0
        import itertools
        for fu in concurrent.futures.as_completed(futs):
            pass
    got = sum(os.path.getsize(f"{OUT}.part{ci:04d}") for ci in range(len(parts))
              if os.path.exists(f"{OUT}.part{ci:04d}"))
    el = time.time() - t0
    print(f"downloaded {got/1024/1024:.1f} MB in {el:.0f}s ({got/1024/el:.0f} KB/s)")
    if got != total:
        raise SystemExit(f"incomplete: {got} != {total}")

    # concatenate
    print("concatenating ...")
    with open(OUT, "wb") as w:
        for ci in range(len(parts)):
            with open(f"{OUT}.part{ci:04d}", "rb") as r:
                while True:
                    b = r.read(4 * 1024 * 1024)
                    if not b:
                        break
                    w.write(b)
    print("size:", os.path.getsize(OUT))
    for ci in range(len(parts)):
        os.remove(f"{OUT}.part{ci:04d}")
    print("DONE")


if __name__ == "__main__":
    main()
