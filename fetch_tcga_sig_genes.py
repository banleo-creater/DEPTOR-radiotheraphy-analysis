#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fetch TCGA-HNSC expression for the union of the EPIC and quanTIseq signature
genes (plus DEPTOR), in ONE cBioPortal batch request.

Why: the earlier TCGA matrix (tcga_hnsc_deconv_genes.csv) contains only the 67
marker genes used by the in-house solver, which overlaps just 4/170 TIL10 genes
and 15/98 EPIC sigGenes.  EPIC/quanTIseq need their own signature genes.

Output: data/tcga_hnsc_sig_genes.csv   (samples x genes, linear RSEM)
"""
import json, os, subprocess, csv
import pandas as pd
import rdata

BASE = r"WorkBuddy/2026-10-17-11-15-46"
DATA = os.path.join(BASE, "data")
CBIO = "http<PATH>"
STUDY = "hnsc_tcga_pan_can_atlas_2018"
PROFILE = f"{STUDY}_rna_seq_v2_mrna"

# ---------------------------------------------------------------- gene universe
til = pd.read_csv(os.path.join(DATA, "signatures", "TIL10_signature.txt"),
                  sep="\t", index_col=0)
epic = rdata.read_rda(os.path.join(DATA, "signatures", "TRef.rda"))["TRef"]
epic_genes = [str(g) for g in epic["sigGenes"]]

GENES = sorted(set(til.index.astype(str)) | set(epic_genes) | {"DEPTOR"})
print(f"signature genes: TIL10={len(til)}  EPIC={len(epic_genes)}  union+DEPTOR={len(GENES)}")


def curl(url, out, method="GET", body=None):
    cmd = ["curl", "-s", "--ssl-no-revoke", "--max-time", "180", "-X", method, url, "-o", out, "-w", "%{http_code}"]
    if body is not None:
        cmd += ["-H", "Content-Type: application/json", "-d", json.dumps(body)]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    return r.stdout.decode().strip()


# ---------------------------------------------------------------- symbol -> entrez
cache_p = os.path.join(DATA, "sig_symbol2entrez.json")
cache = {}
if os.path.exists(cache_p):
    cache = json.load(open(cache_p))

sym2ent = {k: int(v) for k, v in cache.items() if v}
todo = [g for g in GENES if g not in cache]
print(f"resolving {len(todo)} new symbols ({len(sym2ent)} cached) ...")

for i, sym in enumerate(todo):
    p = os.path.join("/tmp", f"_g_{sym}.json")
    curl(f"{CBIO}/genes/{sym}", p)
    try:
        d = json.load(open(p))
        if isinstance(d, dict) and d.get("entrezGeneId"):
            sym2ent[sym] = int(d["entrezGeneId"])
            cache[sym] = int(d["entrezGeneId"])
        else:
            cache[sym] = None
    except Exception:
        cache[sym] = None
    if (i + 1) % 40 == 0:
        print(f"  {i+1}/{len(todo)}")
json.dump(cache, open(cache_p, "w"))

missing = [g for g in GENES if g not in sym2ent]
print(f"resolved {len(sym2ent)}/{len(GENES)}; unresolved ({len(missing)}): {missing[:25]}")

ents = sorted(set(sym2ent.values()))

# ---------------------------------------------------------------- sample ids + fetch
p_sid = os.path.join(DATA, "sig_sids.json")
curl(f"{CBIO}/sample-lists/{PROFILE}/sample-ids", p_sid)
sids = json.load(open(p_sid))
print(f"samples: {len(sids)}")

# batch the fetch: 250 genes x 515 samples is ~20 MB, which gets truncated
BATCH = 50
expr = []
for b0 in range(0, len(ents), BATCH):
    chunk = ents[b0:b0 + BATCH]
    p_exp = os.path.join("/tmp", f"sig_expr_{b0}.json")
    ok = False
    for attempt in range(4):
        code = curl(f"{CBIO}/molecular-profiles/{PROFILE}/molecular-data/fetch",
                    p_exp, "POST",
                    {"entrezGeneIds": chunk, "sampleIds": sids})
        try:
            part = json.load(open(p_exp))
            expr.extend(part)
            print(f"  batch {b0//BATCH+1}: {len(chunk)} genes -> {len(part)} records (http={code})")
            ok = True
            break
        except Exception as e:
            print(f"  batch {b0//BATCH+1} retry {attempt+1}: {str(e)[:60]}")
    if not ok:
        print(f"  batch {b0//BATCH+1} FAILED")
print(f"records total: {len(expr)}")

# ---------------------------------------------------------------- assemble
ent2sym = {}
for s, e in sym2ent.items():
    ent2sym.setdefault(e, s)

rows = {}
for rec in expr:
    e = rec.get("entrezGeneId") or rec.get("geneticEntityId")
    try:
        e = int(e)
    except Exception:
        continue
    s = ent2sym.get(e)
    if s:
        rows.setdefault(rec["sampleId"], {})[s] = rec.get("value")

out = os.path.join(DATA, "tcga_hnsc_sig_genes.csv")
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["sampleId"] + GENES)
    w.writeheader()
    for sid, d in rows.items():
        r = {"sampleId": sid}
        r.update({g: d.get(g) for g in GENES})
        w.writerow(r)

print(f"saved {len(rows)} samples x {len(GENES)} genes -> {out}")
present = [g for g in GENES if any(g in d for d in rows.values())]
print(f"genes with data: {len(present)}/{len(GENES)}")
