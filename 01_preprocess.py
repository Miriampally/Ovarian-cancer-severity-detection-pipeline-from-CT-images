"""Step 1 - Scan TCIA DICOMs, pick one CT series per patient, preprocess, cache as uint8 .npy

Output (work_dir):
  series_inventory.csv   every series found
  series_selected.csv    best-ranked series per patient
  series_used.csv        series actually used (differs when a fallback was needed)
  cache/<patient>.npy    uint8 volume (Z, H, W), soft-tissue windowed, body-cropped, square
  qc/<patient>.png       montage for visual quality control
"""
import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
import torch
import torch.nn.functional as F
import yaml
from scipy import ndimage as ndi
from tqdm import tqdm


def scan_dicoms(root):
    reader = sitk.ImageSeriesReader()
    rows = []
    dirs = [dp for dp, _, fs in os.walk(root) if any(f.lower().endswith(".dcm") for f in fs)]
    for dp in tqdm(dirs, desc="Scanning folders"):
        for sid in reader.GetGDCMSeriesIDs(dp):
            files = reader.GetGDCMSeriesFileNames(dp, sid)
            try:
                ds = pydicom.dcmread(files[0], stop_before_pixels=True)
            except Exception as e:
                print(f"[warn] cannot read {files[0]}: {e}")
                continue
            try:
                thick = float(ds.get("SliceThickness", np.nan))
            except (TypeError, ValueError):
                thick = np.nan
            rows.append(dict(
                patient_id=str(ds.get("PatientID", "")).strip(),
                study_date=str(ds.get("StudyDate", "")),
                series_uid=sid, dir=dp, n_slices=len(files),
                modality=str(ds.get("Modality", "")),
                thickness=thick,
                description=str(ds.get("SeriesDescription", "")),
                image_type="\\".join(map(str, ds.get("ImageType", []))),
                kernel=str(ds.get("ConvolutionKernel", "")),
            ))
    return pd.DataFrame(rows)


def rank_series(inv, cfg):
    """Per patient: all candidate series, best first (baseline study, soft kernel, most slices)."""
    df = inv[(inv.modality == "CT")
             & (~inv.image_type.str.upper().str.contains("LOCALIZER"))
             & (inv.n_slices >= cfg["min_slices"])].copy()
    df["thickness"] = df.thickness.fillna(cfg["max_thickness"])
    df = df[df.thickness <= cfg["max_thickness"]]
    txt = (df.kernel + " " + df.description).str.upper()
    df["bad_kernel"] = txt.str.contains("BONE|LUNG|SHARP")
    df = df.sort_values(["patient_id", "study_date", "bad_kernel", "n_slices"],
                        ascending=[True, True, True, False])
    return {pid: g.reset_index(drop=True) for pid, g in df.groupby("patient_id")}


def resample(img, spacing):
    osp, osz = img.GetSpacing(), img.GetSize()
    nsz = [max(1, int(round(s * o / n))) for s, o, n in zip(osz, osp, spacing)]
    r = sitk.ResampleImageFilter()
    r.SetOutputSpacing(spacing); r.SetSize(nsz)
    r.SetOutputDirection(img.GetDirection()); r.SetOutputOrigin(img.GetOrigin())
    r.SetInterpolator(sitk.sitkLinear); r.SetDefaultPixelValue(-1024)
    return r.Execute(img)


def body_bbox(arr):
    """Heuristic body box from a projection of voxels > -300 HU (removes air around patient)."""
    proj = (arr > -300).mean(0) > 0.5
    proj = ndi.binary_opening(proj, iterations=3)
    lab, n = ndi.label(proj)
    if n == 0:
        return 0, arr.shape[1], 0, arr.shape[2]
    k = int(np.argmax(ndi.sum(proj, lab, range(1, n + 1)))) + 1
    ys, xs = np.where(lab == k)
    return ys.min(), ys.max() + 1, xs.min(), xs.max() + 1


def preprocess_volume(row, cfg):
    sitk.ProcessObject.SetGlobalWarningDisplay(False)   # geometry is audited by 01b_check_slices.py
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(row["dir"], row["series_uid"]))
    img = sitk.Cast(reader.Execute(), sitk.sitkFloat32)       # rescale slope/intercept already applied -> HU
    d = np.array(img.GetDirection()).reshape(3, 3)
    if abs(d[2, 2]) < 0.9:
        raise ValueError("non-axial series")
    img = resample(img, (cfg["spacing_xy"], cfg["spacing_xy"], cfg["spacing_z"]))
    arr = sitk.GetArrayFromImage(img)                         # (Z, Y, X)

    y0, y1, x0, x1 = body_bbox(arr)
    arr = arr[:, y0:y1, x0:x1]
    lo, hi = cfg["window"]
    arr = np.clip((arr - lo) / (hi - lo), 0, 1)
    h, w = arr.shape[1:]
    s = max(h, w)                                             # pad to square
    arr = np.pad(arr, ((0, 0), ((s - h) // 2, s - h - (s - h) // 2), ((s - w) // 2, s - w - (s - w) // 2)))
    t = torch.from_numpy(arr.astype(np.float32)).unsqueeze(1)
    t = F.interpolate(t, size=(cfg["image_size"],) * 2, mode="bilinear", antialias=True, align_corners=False)
    return (t.squeeze(1).clamp(0, 1).numpy() * 255).astype(np.uint8)


def save_qc(vol, path, n=8):
    idx = np.linspace(0, len(vol) - 1, n).round().astype(int)
    plt.imsave(path, np.concatenate([vol[i] for i in idx], axis=1), cmap="gray", vmin=0, vmax=255)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--rescan", action="store_true", help="ignore existing inventory csv")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    work = Path(cfg["paths"]["work_dir"])
    (work / "cache").mkdir(parents=True, exist_ok=True)
    (work / "qc").mkdir(exist_ok=True)

    inv_path = work / "series_inventory.csv"
    if inv_path.exists() and not args.rescan:
        inv = pd.read_csv(inv_path, dtype={"study_date": str})
        inv = inv.fillna({"description": "", "image_type": "", "kernel": ""})
    else:
        inv = scan_dicoms(cfg["paths"]["dicom_root"])
        inv.to_csv(inv_path, index=False)
    print(f"{len(inv)} series, {inv.patient_id.nunique()} patients found")

    cands = rank_series(inv, cfg["preprocess"])
    sel = pd.DataFrame([g.iloc[0] for g in cands.values()]).reset_index(drop=True)
    sel.to_csv(work / "series_selected.csv", index=False)
    print(f"{len(sel)} patients with a usable CT series")

    used, failed = {pid: g.iloc[0] for pid, g in cands.items()}, []
    for pid, g in tqdm(cands.items(), desc="Preprocessing"):
        out = work / "cache" / f"{pid}.npy"
        if out.exists():
            continue
        err = "no candidate series"
        for k, (_, row) in enumerate(g.head(5).iterrows()):       # try best series first, then fallbacks
            try:
                vol = preprocess_volume(row, cfg["preprocess"])
                np.save(out, vol)
                save_qc(vol, work / "qc" / f"{pid}.png")
                used[pid] = row
                err = None
                if k > 0:
                    print(f"[info] {pid}: used fallback series #{k} ({row['description']})")
                break
            except Exception as e:
                err = str(e)
        if err is not None:
            failed.append((pid, err))
            print(f"[fail] {pid}: {err}")

    pd.DataFrame([r for p, r in used.items() if (work / "cache" / f"{p}.npy").exists()]).to_csv(
        work / "series_used.csv", index=False)
    fpath = work / "preprocess_failed.csv"
    if failed:
        pd.DataFrame(failed, columns=["patient_id", "error"]).to_csv(fpath, index=False)
    elif fpath.exists():
        fpath.unlink()
    print(f"Cached volumes: {len(list((work / 'cache').glob('*.npy')))}")
    print("Done. Check work/qc/*.png and work/series_used.csv, then run 01b_check_slices.py")


if __name__ == "__main__":
    main()
