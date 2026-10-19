#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Selective download of GSE164690 CD45n (tumor epithelium + stroma) 10x samples only,
via byte-offset ranges inside the GEO RAW tar. Reuses curl (robust here)."""
import os, re, subprocess, time

BASE = "WorkBuddy/2026-10-17-11-15-46"
TAR_URL = ("http<PATH>"
           "GSE164690/suppl/GSE164690_RAW.tar")
RAW = os.path.join(BASE, "data", "raw")
os.makedirs(RAW, exist_ok=True)

SUFFIX_RE = re.compile(r"_(barcodes|features|matrix)\.(tsv|mtx)\.gz$")
def sample_of(name): return SUFFIX_RE.sub("", name)

members = []
with open(os.path.join(BASE, "data", "filelist.txt")) as f:
    next(f)
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 5: continue
        if parts[0].strip() == "File":
            members.append((parts[1], int(parts[3])))

def read_first_header():
    out = subprocess.run(["curl","-s","--ssl-no-revoke","-r","0-511",TAR_URL], stdout=subprocess.PIPE)
    return out.stdout[:512]

hdr = read_first_header()
name_field = hdr[0:100].split(b"\x00")[0].decode("utf-8","ignore")
first_name = members[0][0]
if name_field and name_field != first_name and first_name in name_field:
    prefix = name_field.rstrip("/") + "/"
    offset = 1024
    members = [(m[0].replace(prefix,"",1) if m[0].startswith(prefix) else m[0], m[1]) for m in members]
    print(f"[dir-entry] prefix='{prefix}' data_offset={offset}")
else:
    offset = 0

ranges = []
cur = offset
for name, size in members:
    data_start = cur + 512
    ranges.append((name, data_start, size))
    cur += 512 + ((size + 511)//512)*512

cd45n = set()
for name,_,_ in ranges:
    if "CD45n" in sample_of(name):
        cd45n.add(sample_of(name))
print(f"Selected {len(cd45n)} CD45n samples: {sorted(cd45n)}", flush=True)

def fetch(name, data_start, size):
    d = os.path.join(RAW, sample_of(name)); os.makedirs(d, exist_ok=True)
    dest = os.path.join(d, name)
    if os.path.exists(dest) and os.path.getsize(dest) == size:
        return "skip"
    end = data_start + size - 1
    for attempt in range(3):
        r = subprocess.run(["curl","-s","--ssl-no-revoke","--retry","2","-r",f"{data_start}-{end}",
                            "-o",dest,TAR_URL])
        if r.returncode == 0 and os.path.exists(dest) and os.path.getsize(dest) == size:
            return "ok"
        time.sleep(2)
    print(f"  FAIL {name}", flush=True); return "err"

total = sum(s for n,_,s in ranges if sample_of(n) in cd45n)
done = 0; started = time.time()
for name, ds, sz in ranges:
    if sample_of(name) not in cd45n: continue
    st = fetch(name, ds, sz)
    if st in ("ok","skip"): done += sz
    el = time.time()-started
    rate = (done/1e6)/el if el>0 else 0
    print(f"  {st:4s} {name}  ({done/1e6:.1f}/{total/1e6:.1f} MB, ~{rate:.2f} MB/s)", flush=True)

print("DONE selective CD45n download.", flush=True)
