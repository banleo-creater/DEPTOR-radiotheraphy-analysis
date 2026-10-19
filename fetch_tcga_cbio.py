#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fetch TCGA-HNSC data from cBioPortal:
  - DEPTOR (entrez 64798) mRNA expression, RNA Seq V2 RSEM, all mRNA samples
  - clinical attributes: OS, DFS, radiation (adjuvant), grade, LVI, age, sex, stage
Uses curl via subprocess (reliable in this environment).
"""
import json, os, subprocess, sys

BASE = "WorkBuddy/2026-10-17-11-15-46/data"
CBIO = "http<PATH>"
PROFILE = "hnsc_tcga_rna_seq_v2_mrna"
GENE_ENTREZ = 64798  # DEPTOR

def curl(url, out, method="GET", body=None):
    cmd = ["curl","-s","--ssl-no-revoke","--max-time","90","-X",method,url,"-o",out,
           "-w","%{http_code}"]
    if body is not None:
        cmd += ["-H","Content-Type: application/json","-d",json.dumps(body)]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    code = r.stdout.decode().strip()
    ok = os.path.exists(out) and os.path.getsize(out) > 0
    return code, (os.path.getsize(out) if ok else 0)

# 1) sample ids
sids_path = os.path.join(BASE,"cbio_sids.json")
sids = json.load(open(sids_path))
print(f"mRNA samples: {len(sids)}")

# 2) expression for DEPTOR
expr_path = os.path.join(BASE,"cbio_deptor_expr.json")
code,size = curl(f"{CBIO}/molecular-profiles/{PROFILE}/molecular-data/fetch", expr_path,
                 "POST", {"entrezGeneIds":[GENE_ENTREZ], "sampleIds": sids})
print(f"expression fetch: http={code} size={size}")
expr = json.load(open(expr_path))
print(f"  records: {len(expr)}")
if expr:
    print("  sample:", expr[0])

# 3) clinical attributes
ATTRS = ["OS_MONTHS","OS_STATUS","DFS_MONTHS","DFS_STATUS",
         "RADIATION_TREATMENT_ADJUVANT","GRADE","ANGIOLYMPHATIC_INVASION",
         "AGE","SEX","AJCC_PATHOLOGIC_TUMOR_STAGE","HISTORY_NEOADJUVANT_TRTYN",
         "PHARMACEUTICAL_TX_ADJUVANT","TUMOR_STATUS","PERINEURAL_INVASION"]
clin = {}
for a in ATTRS:
    p = os.path.join(BASE, f"cbio_clin_{a}.json")
    code,size = curl(f"{CBIO}/studies/hnsc_tcga/clinical-data?clinicalDataType=PATIENT&attributeId={a}&projection=SUMMARY", p)
    try:
        recs = json.load(open(p))
    except Exception:
        recs = []
    d = {}
    for r in recs:
        d[r["patientId"]] = r["value"]
    clin[a] = d
    print(f"  clinical {a}: http={code} n={len(d)}  distinct={sorted(set(list(d.values())[:6]))}")

# 4) merge to a tidy CSV
rows = []
for rec in expr:
    pid = rec["patientId"]; sid = rec["sampleId"]
    row = {"patientId": pid, "sampleId": sid, "DEPTOR": rec.get("value")}
    for a in ATTRS:
        row[a] = clin[a].get(pid)
    rows.append(row)

out_csv = os.path.join(BASE,"tcga_hnsc_deptor.csv")
import csv
with open(out_csv,"w",newline="",encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print(f"\nSaved {len(rows)} rows -> {out_csv}")
print("Done.")
