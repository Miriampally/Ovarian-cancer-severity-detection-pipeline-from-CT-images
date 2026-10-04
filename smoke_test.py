"""Smoke test: fake patients + fake features -> runs 03b and 04 end to end in about a minute on CPU.

Usage:  python smoke_test.py
Passes if both methods print fold AUCs well above 0.5 and metrics.json files are written.
Does NOT touch your real data (uses ./work_smoke). Does not need DICOMs or internet.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

root = Path("work_smoke")
(root / "cache").mkdir(parents=True, exist_ok=True)
fdir = root / "features" / "resnet18"
fdir.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(0)
rows = []
for i in range(30):
    pid, y = f"TCGA-XX-{i:04d}", i % 2
    Z = int(rng.integers(30, 60))
    np.save(root / "cache" / f"{pid}.npy", rng.integers(0, 255, (Z, 64, 64), dtype=np.uint8))
    f = rng.normal(size=(Z, 512)).astype(np.float32)
    if y:                                           # weak signal in a subset of slices
        sel = rng.choice(Z, Z // 3, replace=False)
        f[sel, :20] += 1.0
    np.save(fdir / f"{pid}.npy", f.astype(np.float16))
    rows.append(dict(patient_id=pid, label=y, raw=y))
pd.DataFrame(rows).to_csv(root / "labels.csv", index=False)

cfg = yaml.safe_load(open("config.yaml"))
cfg["paths"]["work_dir"] = str(root)
cfg["train"].update(n_splits=3, n_repeats=2, val_fraction=0.2)
cfg["feature_mil"].update(epochs=15, patience=5)
cfg["features"]["num_threads"] = 2
yaml.safe_dump(cfg, open("config_smoke.yaml", "w"))

def run(cmd):
    print("\n$", " ".join(cmd))
    subprocess.run([sys.executable] + cmd, check=True)

run(["03b_train_features_cv.py", "--config", "config_smoke.yaml"])
for m in ["logreg_mean", "attn_mil"]:
    run(["04_evaluate.py", "--config", "config_smoke.yaml", "--results", str(root / "results" / m)])
print("\nSMOKE TEST PASSED - the pipeline runs end to end.")
