#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fetch TCGA-HNSC (PanCancer Atlas 2018) expression of radiotherapy-sensitivity
genes, to support an in-silico ("virtual knockout") analysis of DEPTOR.

Rationale: DEPTOR is a negative regulator of mTORC1. A virtual knockout is
approximated by the DEPTOR-LOW end of the expression spectrum; we then ask what
happens to the pathways that determine radiosensitivity:
  DDR (DNA damage repair), proliferation/cell-cycle, apoptosis, hypoxia, mTORC1.
If DEPTOR-high tumours are low-proliferation / low-DDR / low-hypoxia, that
explains why DEPTOR-high patients benefit more from radiotherapy.
"""
import json, os, subprocess, csv

BASE = "WorkBuddy/2026-10-17-11-15-46/data"
CBIO = "http<PATH>"
STUDY = "hnsc_tcga_pan_can_atlas_2018"
PROFILE = f"{STUDY}_rna_seq_v2_mrna"

GENES = {
 "DDR": ["ATM","ATR","BRCA1","BRCA2","RAD51","RAD51B","RAD51C","RAD51D","CHEK1","CHEK2",
         "MDC1","MRE11","NBN","PRKDC","XRCC4","XRCC5","XRCC6","PARP1","FANCD2","FANCA",
         "PALB2","BRIP1","MLH1","MSH2","MSH6","PMS2"],
 "PROLIF": ["CDK1","CDK2","CDK4","CDK6","CCNB1","CCNE1","CCNA2","PLK1","AURKA","AURKB",
            "WEE1","MKI67","PCNA","TOP2A","BUB1","BUB1B","MAD2L1"],
 "APOP": ["BAX","BAK1","CASP3","CASP8","CASP9","TP53","BBC3","BID","PMAIP1","APAF1"],
 "ANTI_APOP": ["BCL2","BCL2L1","MCL1","XIAP","BIRC5"],
 "HYPOXIA": ["HIF1A","VEGFA","CA9","SLC2A1","LDHA","PGK1","ADM"],
 "MTORC1": ["MTOR","RPTOR","RICTOR","RHEB","TSC1","TSC2","RPS6KB1","EIF4EBP1","AKT1","AKT2",
            "PIK3CA","PIK3R1","MLST8","DEPTOR"],
}
ALL = sorted({g for v in GENES.values() for g in v})

def curl(url, out, method="GET", body=None):
    cmd = ["curl","-s","--ssl-no-revoke","--max-time","120","-X",method,url,"-o",out,"-w","%{http_code}"]
    if body is not None:
        cmd += ["-H","Content-Type: application/json","-d",json.dumps(body)]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    return r.stdout.decode().strip()

# 1) symbol -> entrez  (GET /genes/{symbol} one by one; /genes/fetch rejected the payload)
sym2ent = {}
for sym in ALL:
    p_g = os.path.join(BASE, f"vko_g_{sym}.json")
    curl(f"{CBIO}/genes/{sym}", p_g)
    try:
        d = json.load(open(p_g))
        if isinstance(d, dict) and d.get("entrezGeneId"):
            sym2ent[sym] = int(d["entrezGeneId"])
    except Exception:
        pass
print(f"resolved {len(sym2ent)}/{len(ALL)} genes")
missing = [g for g in ALL if g not in sym2ent]
if missing: print("  unresolved:", missing)
ents = sorted(set(sym2ent.values()))

# 2) sample ids (RNA-seq samples)
p_sid = os.path.join(BASE, "vko_sids.json")
curl(f"{CBIO}/sample-lists/{PROFILE}/sample-ids", p_sid)
sids = json.load(open(p_sid))
print(f"mRNA samples: {len(sids)}")

# 3) fetch expression for all genes
p_exp = os.path.join(BASE, "vko_expr.json")
code = curl(f"{CBIO}/molecular-profiles/{PROFILE}/molecular-data/fetch", p_exp, "POST",
            {"entrezGeneIds": ents, "sampleIds": sids})
expr = json.load(open(p_exp))
print(f"expression records: {len(expr)} (http={code})")

ent2sym = {v: k for k, v in sym2ent.items() if v}
rows = {}
for rec in expr:
    sid = rec["sampleId"]; ent = rec.get("entrezGeneId") or rec.get("geneticEntityId")
    try: ent = int(ent)
    except Exception: pass
    sym = ent2sym.get(ent)
    if not sym: continue
    rows.setdefault(sid, {})[sym] = rec.get("value")

out_csv = os.path.join(BASE, "tcga_hnsc_vko_genes.csv")
allcols = ["sampleId"] + ALL
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=allcols)
    w.writeheader()
    for sid, d in rows.items():
        rec = {"sampleId": sid}
        rec.update({g: d.get(g) for g in ALL})
        w.writerow(rec)
print(f"Saved {len(rows)} samples x {len(ALL)} genes -> {out_csv}")
