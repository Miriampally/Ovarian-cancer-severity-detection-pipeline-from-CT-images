"""Step 1b - Audit slice geometry of the series actually used (reads DICOM headers only, fast).

Finds the patients behind SimpleITK's "Non uniform sampling or missing slices" warnings.

Output: work_dir/series_zqc.csv
  median_spacing_mm : typical distance between neighbouring slices
  max_dev_mm        : biggest deviation of any gap from that typical spacing
  dup_frac          : fraction of slices at (almost) the same position as the previous one
  severe            : 1 if max_dev_mm > 20 or dup_frac > 0.05  (merged/broken series -> excluded by 02_labels.py)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import yaml


def z_stats(positions):
    """positions: slice coordinates (mm) along the slice normal."""
    p = np.sort(np.asarray(positions, dtype=float))
    d = np.diff(p)
    if len(d) == 0:
        return dict(median_spacing_mm=np.nan, max_dev_mm=np.nan, dup_frac=np.nan, extent_mm=0.0)
    nz = d[d > 0.01]
    med = float(np.median(nz)) if len(nz) else 0.0
    return dict(median_spacing_mm=round(med, 3),
                max_dev_mm=round(float(np.max(np.abs(d - med))), 3),
                dup_frac=round(float(np.mean(d <= 0.01)), 4),
                extent_mm=round(float(p[-1] - p[0]), 1))


def series_positions(dirpath, series_uid):
    import SimpleITK as sitk
    files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(dirpath), str(series_uid))
    pos = []
    for f in files:
        ds = pydicom.dcmread(f, stop_before_pixels=True)
        ipp = np.array(ds.ImagePositionPatient, dtype=float)
        iop = np.array(ds.ImageOrientationPatient, dtype=float)
        normal = np.cross(iop[:3], iop[3:])
        pos.append(float(ipp @ normal))
    return pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--max_dev", type=float, default=20.0)
    ap.add_argument("--max_dup", type=float, default=0.05)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    work = Path(cfg["paths"]["work_dir"])

    used = pd.read_csv(work / "series_used.csv")
    rows = []
    for _, r in used.iterrows():
        try:
            st = z_stats(series_positions(r["dir"], r["series_uid"]))
        except Exception as e:
            st = dict(median_spacing_mm=np.nan, max_dev_mm=np.nan, dup_frac=np.nan, extent_mm=np.nan)
            print(f"[warn] {r['patient_id']}: {e}")
        rows.append(dict(patient_id=r["patient_id"], n_slices=r["n_slices"], **st))
    q = pd.DataFrame(rows)
    q["severe"] = ((q.max_dev_mm > args.max_dev) | (q.dup_frac > args.max_dup)).astype(int)
    q.to_csv(work / "series_zqc.csv", index=False)

    print(f"{len(q)} patients checked")
    print("Median slice spacing (mm), counts:\n", q.median_spacing_mm.round(1).value_counts().sort_index().to_string())
    print(f"\nSevere geometry problems: {int(q.severe.sum())} patients")
    if q.severe.sum():
        print(q[q.severe == 1][["patient_id", "n_slices", "median_spacing_mm", "max_dev_mm", "dup_frac"]].to_string(index=False))
    print("\nSaved work/series_zqc.csv. 02_labels.py will drop 'severe' patients automatically.")


if __name__ == "__main__":
    main()
