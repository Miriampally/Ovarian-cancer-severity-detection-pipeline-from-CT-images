"""How did the geometry-flagged patients fare in the sensitivity run?

For each model prints, for flagged patients that have a label: their true label, mean predicted
probability of death <=24 months (averaged over CV repeats) and where that score ranks among ALL
patients (0 = lowest risk score, 1 = highest). Positives with a LOW rank or negatives with a HIGH
rank are being scored wrongly.  With ~6 patients this is an observation, not proof.

Usage:  python check_flagged.py --config config_geometry.yaml
"""
import argparse
from pathlib import Path

import pandas as pd
import yaml

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="config_geometry.yaml")
args = ap.parse_args()
cfg = yaml.safe_load(open(args.config))
work, rdir = Path(cfg["paths"]["work_dir"]), cfg["paths"].get("results_dir", "results")

bad = set(pd.read_csv(work / "series_zqc.csv").query("severe == 1").patient_id)
for method in ["logreg_mean", "attn_mil"]:
    o = pd.read_csv(work / rdir / method / "oof_predictions.csv")
    per = o.groupby("patient_id")[["label", "p1"]].mean()
    per["score_rank"] = per.p1.rank(pct=True)
    sub = per[per.index.isin(bad)].round(2)
    print(f"\n=== {method}: {len(sub)} flagged patients with a label (of {len(per)} total) ===")
    print(sub.sort_values("label", ascending=False).to_string())
    print(f"reference - median score of all positives: {per[per.label == 1].p1.median():.2f}, "
          f"of all negatives: {per[per.label == 0].p1.median():.2f}")
