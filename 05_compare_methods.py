"""Step 5 - Thesis-ready comparison of methods (run 04_evaluate.py for each method first).

Output (work_dir/results): comparison.csv, comparison.md
Includes a paired patient-level bootstrap of the AUC difference between each method and the first one
(binary labels only). Report it as exploratory: with ~100 patients, differences are rarely significant.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score


def per_patient(path):
    oof = pd.read_csv(path)
    pc = [c for c in oof.columns if c.startswith("p") and c[1:].isdigit()]
    return oof.groupby("patient_id").agg(label=("label", "first"), **{c: (c, "mean") for c in pc}).sort_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--methods", nargs="+", default=["logreg_mean", "attn_mil"], help="first = reference")
    ap.add_argument("--n_boot", type=int, default=2000)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    res = Path(cfg["paths"]["work_dir"]) / cfg["paths"].get("results_dir", "results")

    rows, preds = [], {}
    for m in args.methods:
        mj = json.load(open(res / m / "metrics.json"))
        preds[m] = per_patient(res / m / "oof_predictions.csv")
        f = lambda k: f"{mj[k]:.3f} [{mj[k + '_ci95'][0]:.3f}-{mj[k + '_ci95'][1]:.3f}]"
        rows.append({"method": m, "n": mj["n_patients"], "AUC [95% CI]": f("auc"),
                     "AUC per repeat (mean±sd)": f"{mj['auc_per_repeat_mean_sd'][0]:.3f}±{mj['auc_per_repeat_mean_sd'][1]:.3f}",
                     "Balanced acc [95% CI]": f("balanced_acc"),
                     "Sensitivity": f"{mj.get('sensitivity', float('nan')):.3f}",
                     "Specificity": f"{mj.get('specificity', float('nan')):.3f}"})
    tab = pd.DataFrame(rows)

    ref = args.methods[0]
    diffs = []
    for m in args.methods[1:]:
        a, b = preds[ref], preds[m]
        ids = a.index.intersection(b.index)
        y = a.loc[ids, "label"].values
        if len(np.unique(y)) != 2:
            continue
        pa, pb = a.loc[ids, "p1"].values, b.loc[ids, "p1"].values
        rng, d = np.random.default_rng(0), []
        for _ in range(args.n_boot):
            i = rng.integers(0, len(y), len(y))
            if len(np.unique(y[i])) < 2:
                continue
            d.append(roc_auc_score(y[i], pb[i]) - roc_auc_score(y[i], pa[i]))
        lo, hi = np.percentile(d, [2.5, 97.5])
        diffs.append(f"AUC({m}) - AUC({ref}) = {roc_auc_score(y, pb) - roc_auc_score(y, pa):+.3f} "
                     f"[95% CI {lo:+.3f}, {hi:+.3f}]")

    tab.to_csv(res / "comparison.csv", index=False)
    try:
        md = tab.to_markdown(index=False)
    except ImportError:                       # tabulate not installed
        md = tab.to_string(index=False)
    text = md + "\n\nPaired bootstrap (patient level):\n" + "\n".join(diffs)
    (res / "comparison.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
