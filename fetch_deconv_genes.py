#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fetch TCGA-HNSC expression of cell-type marker genes + hypoxia genes, for
  (a) NNLS deconvolution of bulk DEPTOR into tumour vs immune contributions
  (b) mediation analysis  DEPTOR -> HIF1A/hypoxia -> radiotherapy benefit
"""
import json, os, subprocess, csv

BASE = "WorkBuddy/2026-10-17-11-15-46/data"
CBIO = "http<PATH>"
STUDY = "hnsc_tcga_pan_can_atlas_2018"
PROFILE = f"{STUDY}_rna_seq_v2_mrna"

SETS = {
 "Tumor":  ["KRT5","KRT6A","KRT14","KRT8","KRT18","EPCAM","TP63","KRT19","SFN","KRT17"],
 "Myeloid":["LYZ","CD68","CD163","CSF1R","CD14","ITGAM","MRC1","VSIG4","AIF1","C1QA","C1QB",
            "FCGR1A","SPI1","LST1","TYROBP"],
 "Tcell":  ["CD3D","CD3E","CD8A","CD4","IL7R","CD2","CD3G","CD28"],
 "Bcell":  ["CD79A","MS4A1","JCHAIN","MZB1","CD19"],
 "NK":     ["NCAM1","NKG7","KLRF1","KLRD1","GNLY","KLRK1","PRF1"],
 "Stroma": ["COL1A1","COL1A2","ACTA2","PDGFRA","DCN","LUM","FAP","PDGFRB"],
 "Endo":   ["PECAM1","VWF","CLDN5","CDH5"],
 "Hypoxia":["HIF1A","CA9","VEGFA","SLC2A1","LDHA","PGK1","ADM","BNIP3","PDK1"],
}
ALL = sorted({g for v in SETS.values() for g in v} | {"DEPTOR"})

def curl(url, out, method="GET", body=None):
    cmd = ["curl","-s","--ssl-no-revoke","--max-time","120","-X",method,url,"-o",out,"-w","%{http_code}"]
    if body is not None:
        cmd += ["-H","Content-Type: application/json","-d",json.dumps(body)]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    return r.stdout.decode().strip()

sym2ent = {}
for sym in ALL:
    p = os.path.join(BASE, f"dec_g_{sym}.json")
    curl(f"{CBIO}/genes/{sym}", p)
    try:
        d = json.load(open(p))
        if isinstance(d, dict) and d.get("entrezGeneId"):
            sym2ent[sym] = int(d["entrezGeneId"])
    except Exception:
        pass
print(f"resolved {len(sym2ent)}/{len(ALL)}")
print("missing:", [g for g in ALL if g not in sym2ent])
ents = sorted(set(sym2ent.values()))

p_sid = os.path.join(BASE, "dec_sids.json")
curl(f"{CBIO}/sample-lists/{PROFILE}/sample-ids", p_sid)
sids = json.load(open(p_sid))

p_exp = os.path.join(BASE, "dec_expr.json")
code = curl(f"{CBIO}/molecular-profiles/{PROFILE}/molecular-data/fetch", p_exp, "POST",
            {"entrezGeneIds": ents, "sampleIds": sids})
expr = json.load(open(p_exp))
print(f"records {len(expr)} (http={code})")

ent2sym = {v: k for k, v in sym2ent.items()}
rows = {}
for rec in expr:
    e = rec.get("entrezGeneId") or rec.get("geneticEntityId")
    try: e = int(e)
    except Exception: pass
    s = ent2sym.get(e)
    if s: rows.setdefault(rec["sampleId"], {})[s] = rec.get("value")

out = os.path.join(BASE, "tcga_hnsc_deconv_genes.csv")
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["sampleId"] + ALL)
    w.writeheader()
    for sid, d in rows.items():
        r = {"sampleId": sid}; r.update({g: d.get(g) for g in ALL}); w.writerow(r)
print(f"saved {len(rows)} x {len(ALL)} -> {out}")
