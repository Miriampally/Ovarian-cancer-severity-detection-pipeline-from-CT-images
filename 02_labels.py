"""Step 2 - Build severity labels from the TCGA clinical table and intersect with preprocessed patients.

Output: work_dir/labels.csv  (patient_id, label, raw)
"""
import argparse
import re
from pathlib import Path

import pandas as pd
import yaml

PATIENT_CANDS = ["patient id", "patient_id", "bcr_patient_barcode", "case_submitter_id", "submitter_id", "case id"]
STAGE_CANDS = ["neoplasm disease stage american joint committee on cancer code", "clinical_stage",
               "figo_stage", "ajcc_pathologic_tumor_stage", "ajcc_pathologic_stage", "pathologic_stage", "stage"]


def find_col(df, given, cands, contains=None):
    if given != "auto":
        return given
    low = {c.lower().strip(): c for c in df.columns}
    for c in cands:
        if c in low:
            return low[c]
    if contains:
        for k, v in low.items():
            if contains in k:
                return v
    raise KeyError(f"Could not auto-detect column among {list(df.columns)[:30]}...; set it in config.yaml")


def parse_stage(s):
    m = re.search(r"STAGE\s+(IV|III|II|I)", str(s).upper())
    return {"I": 1, "II": 2, "III": 3, "IV": 4}[m.group(1)] if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    lc, work = cfg["labels"], Path(cfg["paths"]["work_dir"])

    clin = pd.read_csv(cfg["paths"]["clinical_file"], sep=None, engine="python", comment="#")
    pcol = find_col(clin, lc["patient_col"], PATIENT_CANDS)
    clin["patient_id"] = clin[pcol].astype(str).str.strip().str[:12]
    clin = clin.drop_duplicates("patient_id")

    mode = lc["mode"]
    if mode.startswith("stage"):
        scol = find_col(clin, lc["stage_col"], STAGE_CANDS, contains="stage")
        clin["stage"] = clin[scol].map(parse_stage)
        clin = clin.dropna(subset=["stage"])
        print("Stage distribution (whole clinical table):\n", clin.stage.value_counts().sort_index())
        if mode == "stage_binary":
            clin["label"] = (clin.stage >= 3).astype(int)
        elif mode == "stage_34":
            clin = clin[clin.stage >= 3]
            clin["label"] = (clin.stage == 4).astype(int)
        elif mode == "stage_3class":
            clin["label"] = clin.stage.map({1: 0, 2: 0, 3: 1, 4: 2})
        clin["raw"] = clin.stage
    elif mode == "os_24m":
        m = pd.to_numeric(clin[lc["os_months_col"]], errors="coerce")
        ev = clin[lc["os_status_col"]].astype(str).str.upper().str.contains("DECEASED|^1|DEAD", regex=True)
        clin["label"] = None
        clin.loc[ev & (m <= 24), "label"] = 1
        clin.loc[m > 24, "label"] = 0
        clin = clin.dropna(subset=["label"])
        clin["label"] = clin.label.astype(int)
        clin["raw"] = m
    else:
        raise ValueError(f"Unknown label mode {mode}")

    have_img = {p.stem for p in (work / "cache").glob("*.npy")}
    zq = work / "series_zqc.csv"
    if lc.get("exclude_severe_geometry", True) and zq.exists():
        bad = set(pd.read_csv(zq).query("severe == 1").patient_id)
        print(f"Excluding {len(have_img & bad)} patients with severe slice-geometry problems (series_zqc.csv)")
        have_img -= bad
    out = clin[clin.patient_id.isin(have_img)][["patient_id", "label", "raw"]].reset_index(drop=True)
    out.to_csv(work / cfg["paths"].get("labels_file", "labels.csv"), index=False)

    print(f"\nPatients with image + label: {len(out)} (images available: {len(have_img)})")
    print("Class counts:\n", out.label.value_counts().sort_index())
    if out.label.value_counts().min() < 15:
        print("\n[WARNING] smallest class has <15 patients - AUC will be very unstable. "
              "Consider another label mode and report CIs.")


if __name__ == "__main__":
    main()
