#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fetch TCGA-HNSC expression for NKG2D-ligand genes and an NK signature, to test:

    myeloid DEPTOR high -> few NK
    radiotherapy up-regulates NKG2D ligands (MICA/MICB/ULBP1-6)
    -> but if NK are already scarce, the ligand up-regulation has "no receiver"
    -> poorer radiotherapy benefit

Genes
-----
NKG2D ligands : MICA, MICB, ULBP1..ULBP6 (a.k.a. RAET1*)
NK receptor   : KLRK1 (NKG2D)
NK signature  : NKG7, GNLY, KLRD1, PRF1, FCGR3A, NCAM1, KLRF1, XCL1, XCL2, TBX21, EOMES
"""
import json, os, subprocess, csv

BASE = "WorkBuddy/2026-10-17-11-15-46/data"
CBIO = "http<PATH>"
STUDY = "hnsc_tcga_pan_can_atlas_2018"
PROFILE = f"{STUDY}_rna_seq_v2_mrna"

NKG2D_LIGANDS = ["MICA","MICB","ULBP1","ULBP2","ULBP3","ULBP4","ULBP5","ULBP6",
                 "RAET1E","RAET1G","RAET1L","RAET1H","RAET1N","RAET1I"]
NK_SIG = ["NKG7","GNLY","KLRD1","PRF1","FCGR3A","NCAM1","KLRF1","XCL1","XCL2","TBX21","EOMES"]
NK_RECEPTOR = ["KLRK1"]
ALL = sorted(set(NKG2D_LIGANDS + NK_SIG + NK_RECEPTOR))

def curl(url, out, method="GET", body=None):
    cmd = ["curl","-s","--ssl-no-revoke","--max-time","120","-X",method,url,"-o",out,"-w","%{http_code}"]
    if body is not None:
        cmd += ["-H","Content-Type: application/json","-d",json.dumps(body)]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    return r.stdout.decode().strip()

sym2ent = {}
for sym in ALL:
    p_g = os.path.join(BASE, f"nkg2d_g_{sym}.json")
    curl(f"{CBIO}/genes/{sym}", p_g)
    try:
        d = json.load(open(p_g))
        if isinstance(d, dict) and d.get("entrezGeneId"):
            sym2ent[sym] = int(d["entrezGeneId"])
    except Exception:
        pass
print(f"resolved {len(sym2ent)}/{len(ALL)}")
print("  resolved:", sorted(sym2ent))
print("  missing :", [g for g in ALL if g not in sym2ent])
ents = sorted(set(sym2ent.values()))

p_sid = os.path.join(BASE, "nkg2d_sids.json")
curl(f"{CBIO}/sample-lists/{PROFILE}/sample-ids", p_sid)
sids = json.load(open(p_sid))
print("mRNA samples:", len(sids))

p_exp = os.path.join(BASE, "nkg2d_expr.json")
code = curl(f"{CBIO}/molecular-profiles/{PROFILE}/molecular-data/fetch", p_exp, "POST",
            {"entrezGeneIds": ents, "sampleIds": sids})
expr = json.load(open(p_exp))
print(f"records: {len(expr)} (http={code})")

ent2sym = {v: k for k, v in sym2ent.items()}
rows = {}
for rec in expr:
    ent = rec.get("entrezGeneId") or rec.get("geneticEntityId")
    try: ent = int(ent)
    except Exception: pass
    sym = ent2sym.get(ent)
    if not sym: continue
    rows.setdefault(rec["sampleId"], {})[sym] = rec.get("value")

out_csv = os.path.join(BASE, "tcga_hnsc_nkg2d_nk.csv")
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["sampleId"] + ALL)
    w.writeheader()
    for sid, d in rows.items():
        rec = {"sampleId": sid}; rec.update({g: d.get(g) for g in ALL})
        w.writerow(rec)
print(f"saved {len(rows)} samples x {len(ALL)} genes -> {out_csv}")
