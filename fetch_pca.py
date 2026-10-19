#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fetch TCGA-HNSC PanCancer Atlas 2018 data from cBioPortal for DEPTOR survival analysis."""
import json, os, subprocess, csv

BASE = "WorkBuddy/2026-10-17-11-15-46/data"
CBIO = "http<PATH>"
STUDY = "hnsc_tcga_pan_can_atlas_2018"
PROFILE = f"{STUDY}_rna_seq_v2_mrna"
GENE_ENTREZ = 64798  # DEPTOR

def curl(url, out, method="GET", body=None):
    cmd = ["curl","-s","--ssl-no-revoke","--max-time","120","-X",method,url,"-o",out,"-w","%{http_code}"]
    if body is not None:
        cmd += ["-H","Content-Type: application/json","-d",json.dumps(body)]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    return r.stdout.decode().strip(), (os.path.getsize(out) if os.path.exists(out) else 0)

# sample ids
p_sid = os.path.join(BASE,"pca_sids.json")
code,size = curl(f"{CBIO}/sample-lists/{PROFILE}/sample-ids", p_sid)
sids = json.load(open(p_sid))
print(f"mRNA samples: {len(sids)} (http={code})")

# expression
p_expr = os.path.join(BASE,"pca_deptor_expr.json")
code,size = curl(f"{CBIO}/molecular-profiles/{PROFILE}/molecular-data/fetch", p_expr, "POST",
                 {"entrezGeneIds":[GENE_ENTREZ], "sampleIds": sids})
expr = json.load(open(p_expr))
print(f"expression records: {len(expr)} (http={code}, size={size})")

# clinical
ATTRS = ["RADIATION_THERAPY","OS_MONTHS","OS_STATUS","PFS_MONTHS","PFS_STATUS",
         "DSS_MONTHS","DSS_STATUS","DFS_MONTHS","DFS_STATUS","GRADE","AGE","SEX",
         "AJCC_PATHOLOGIC_TUMOR_STAGE","SUBTYPE","HISTORY_NEOADJUVANT_TRTYN",
         "PERSON_NEOPLASM_CANCER_STATUS","NEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT"]
clin = {}
for a in ATTRS:
    p = os.path.join(BASE, f"pca_clin_{a}.json")
    code,size = curl(f"{CBIO}/studies/{STUDY}/clinical-data?clinicalDataType=PATIENT&attributeId={a}&projection=SUMMARY", p)
    try: recs = json.load(open(p))
    except Exception: recs = []
    clin[a] = {r["patientId"]: r["value"] for r in recs}
    print(f"  {a}: n={len(clin[a])}")

rows = []
for rec in expr:
    pid, sid = rec["patientId"], rec["sampleId"]
    row = {"patientId": pid, "sampleId": sid, "DEPTOR_RSEM": rec.get("value")}
    for a in ATTRS: row[a] = clin[a].get(pid)
    rows.append(row)

out_csv = os.path.join(BASE,"tcga_hnsc_pca_deptor.csv")
with open(out_csv,"w",newline="",encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print(f"\nSaved {len(rows)} rows -> {out_csv}")
