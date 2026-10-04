"""Fetch TCGA-OV clinical data (stage, age, survival) from the public GDC API -> gdc_clinical.tsv

No login, no extra packages. Run:   python fetch_gdc_clinical.py
Then:                               python run_all.py --name stage --clinical gdc_clinical.tsv --steps 02

Columns written
  patient_id   TCGA barcode (12 chars)            stage_best   FIGO > AJCC > tumor_stage, normalised to 'STAGE IIIC' style
  age          years at diagnosis                  OS_STATUS    '1:DECEASED' / '0:LIVING'
  OS_MONTHS    months to death or last follow-up   raw_figo, raw_ajcc, raw_tumor   original values for checking
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

API = "https://api.gdc.cancer.gov/cases"
BASE = ["submitter_id", "demographic.vital_status", "demographic.days_to_death", "diagnoses.age_at_diagnosis",
        "diagnoses.days_to_last_follow_up", "diagnoses.tumor_stage"]
EXTRA = ["diagnoses.figo_stage", "diagnoses.ajcc_pathologic_stage"]
EMPTY = {"", "NOT REPORTED", "UNKNOWN", "NA", "N/A", "[NOT AVAILABLE]", "[UNKNOWN]", "NOT ALLOWED TO COLLECT", "'--"}


def query(fields, project, timeout=90, tries=4):
    """GET with retries. Server-side errors (429/5xx) are retried with a pause; HTTP 400 is raised at once."""
    filt = {"op": "in", "content": {"field": "project.project_id", "value": [project]}}
    url = API + "?" + urllib.parse.urlencode({"filters": json.dumps(filt), "fields": ",".join(fields),
                                               "size": "2000", "format": "JSON"})
    req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "ovca-pipeline/1.0"})
    for attempt in range(1, tries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)["data"]["hits"]
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < tries:
                wait = 5 * attempt * attempt
                print(f"[note] GDC returned HTTP {e.code} (their side); retry {attempt}/{tries - 1} in {wait}s")
                time.sleep(wait)
                continue
            raise


def norm_stage(v):
    if v is None:
        return None
    t = str(v).strip().upper()
    if t in EMPTY:
        return None
    return t if t.startswith("STAGE") else "STAGE " + t


def flatten(hits):
    rows = []
    for h in hits:
        dx = (h.get("diagnoses") or [{}])[0]
        dm = h.get("demographic") or {}
        f, a, t = dx.get("figo_stage"), dx.get("ajcc_pathologic_stage"), dx.get("tumor_stage")
        best = next((n for n in (norm_stage(f), norm_stage(a), norm_stage(t)) if n), None)
        vs = dm.get("vital_status")
        dead = str(vs).lower() == "dead"
        days = dm.get("days_to_death") if dead else dx.get("days_to_last_follow_up")
        age_d = dx.get("age_at_diagnosis")
        rows.append(dict(
            patient_id=h.get("submitter_id"), stage_best=best,
            age=(age_d / 365.25 if age_d is not None else None),
            OS_STATUS=("1:DECEASED" if dead else ("0:LIVING" if vs else None)),
            OS_MONTHS=(days / 30.4375 if days is not None else None),
            raw_figo=f, raw_ajcc=a, raw_tumor=t))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="TCGA-OV")
    ap.add_argument("--out", default="gdc_clinical.tsv")
    args = ap.parse_args()
    try:
        try:
            hits = query(BASE + EXTRA, args.project)
        except urllib.error.HTTPError as e:
            if e.code != 400:
                raise
            print("[note] extended field set rejected (HTTP 400); retrying with basic fields")
            hits = query(BASE, args.project)
    except urllib.error.HTTPError as e:
        sys.exit(f"GDC API answered HTTP {e.code} ({e.reason}) after retries. 5xx codes mean the GDC service is "
                 "temporarily unavailable on THEIR side: try again in a while, or download the clinical table "
                 "from the web portal (https://portal.gdc.cancer.gov/projects/TCGA-OV) and send me its header line.")
    except (urllib.error.URLError, TimeoutError) as e:
        sys.exit(f"Could not reach the GDC API (network problem?): {e}")
    df = flatten(hits)
    df.to_csv(args.out, sep="\t", index=False)

    print(f"{len(df)} cases saved to {args.out}")
    print("\nstage_best (what the pipeline will use):")
    print(df.stage_best.value_counts(dropna=False).to_string())
    for c in ("raw_figo", "raw_ajcc", "raw_tumor"):
        print(f"  {c}: {int(df[c].notna().sum())} non-empty")
    print(f"\nage known: {int(df.age.notna().sum())} / {len(df)}")
    print(f"OS_STATUS:\n{df.OS_STATUS.value_counts(dropna=False).to_string()}")


if __name__ == "__main__":
    main()
