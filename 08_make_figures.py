"""Step 8 - Thesis figures and a summary table from whatever analyses have been run.

Looks for these results folders under work_dir (missing ones are skipped):
  results                 main analysis (ResNet-18, flagged series excluded)
  results_geometry_incl   sensitivity: flagged series included
  results_resnet50        sensitivity: ResNet-50 features
  results_stage           secondary label: stage III vs IV
Each must contain  logreg_mean/  and  attn_mil/  with metrics.json (from 04_evaluate.py).

Output (work_dir/figures):
  fig_forest_all_analyses.png    AUC with 95% CI for every analysis x model
  fig_roc_main.png               ROC curves of the main analysis
  summary_table.csv / .md        all numbers in one table
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_curve

ANALYSES = [("results", "Main: ResNet-18, 24-month mortality"),
            ("results_geometry_incl", "Sensitivity: flagged series included"),
            ("results_resnet50", "Sensitivity: ResNet-50 features"),
            ("results_stage", "Secondary label: stage III vs IV")]
METHODS = [("logreg_mean", "Mean-pool logistic regression", "o", "#1f77b4"),
           ("attn_mil", "Attention MIL", "s", "#d62728")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    work = Path(cfg["paths"]["work_dir"])
    out = work / "figures"
    out.mkdir(exist_ok=True)

    rows = []
    for folder, title in ANALYSES:
        for m, mname, _, _ in METHODS:
            f = work / folder / m / "metrics.json"
            if not f.exists():
                continue
            j = json.load(open(f))
            rows.append(dict(analysis=title, folder=folder, method=m, model=mname, n=j["n_patients"],
                             positives=j["class_counts"].get("1", np.nan), auc=j["auc"],
                             auc_lo=j["auc_ci95"][0], auc_hi=j["auc_ci95"][1],
                             auc_per_repeat=f"{j['auc_per_repeat_mean_sd'][0]:.3f}±{j['auc_per_repeat_mean_sd'][1]:.3f}",
                             avg_precision=j.get("average_precision", np.nan),
                             balanced_acc=j["balanced_acc"]))
    if not rows:
        raise SystemExit("No metrics.json found. Run 04_evaluate.py for each method first.")
    df = pd.DataFrame(rows)
    df.to_csv(out / "summary_table.csv", index=False)
    show = df.assign(**{"AUC (95% CI)": df.apply(lambda r: f"{r.auc:.3f} ({r.auc_lo:.3f}-{r.auc_hi:.3f})", axis=1),
                        "n (pos)": df.apply(lambda r: f"{int(r.n)} ({int(r.positives)})", axis=1),
                        "AP": df.avg_precision.round(3), "Bal. acc": df.balanced_acc.round(3)}
                     )[["analysis", "model", "n (pos)", "AUC (95% CI)", "auc_per_repeat", "AP", "Bal. acc"]]
    try:
        md = show.to_markdown(index=False)
    except ImportError:
        md = show.to_string(index=False)
    (out / "summary_table.md").write_text(md)

    # ---- forest plot ---------------------------------------------------------
    present = [a for a in ANALYSES if a[0] in set(df.folder)]
    fig, ax = plt.subplots(figsize=(8, 1.2 + 1.1 * len(present)))
    yt, yl = [], []
    for i, (folder, title) in enumerate(present):
        y0 = len(present) - i
        for k, (m, mname, mk, col) in enumerate(METHODS):
            r = df[(df.folder == folder) & (df.method == m)]
            if r.empty:
                continue
            r, y = r.iloc[0], y0 + (0.15 if k == 0 else -0.15)
            ax.errorbar(r.auc, y, xerr=[[r.auc - r.auc_lo], [r.auc_hi - r.auc]], fmt=mk, color=col, capsize=3,
                        label=mname if i == 0 else None)
        yt.append(y0)
        r0 = df[df.folder == folder].iloc[0]
        yl.append(f"{title}\n(n={int(r0.n)}, {int(r0.positives)} positive)")
    ax.axvline(0.5, color="grey", ls="--", lw=1)
    ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=8)
    ax.set_xlim(0.4, 1.0); ax.set_xlabel("AUC (95% bootstrap CI)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=8, frameon=False)
    plt.tight_layout(); plt.savefig(out / "fig_forest_all_analyses.png", dpi=300, bbox_inches="tight"); plt.close()

    # ---- ROC of main analysis ------------------------------------------------
    plt.figure(figsize=(4.6, 4.6))
    drawn = False
    for m, mname, _, col in METHODS:
        f = work / "results" / m / "oof_predictions.csv"
        if not f.exists():
            continue
        o = pd.read_csv(f)
        per = o.groupby("patient_id").agg(label=("label", "first"), p1=("p1", "mean"))
        fpr, tpr, _ = roc_curve(per.label, per.p1)
        r = df[(df.folder == "results") & (df.method == m)]
        lab = f"{mname}" + (f" (AUC {r.iloc[0].auc:.2f}, {r.iloc[0].auc_lo:.2f}-{r.iloc[0].auc_hi:.2f})" if not r.empty else "")
        plt.plot(fpr, tpr, color=col, lw=2, label=lab); drawn = True
    if drawn:
        plt.plot([0, 1], [0, 1], "k--", lw=1)
        plt.xlabel("1 - Specificity"); plt.ylabel("Sensitivity"); plt.legend(loc="lower right", fontsize=7, frameon=False)
        plt.tight_layout(); plt.savefig(out / "fig_roc_main.png", dpi=300)
    plt.close()

    print(md)
    print(f"\nSaved to {out}:")
    for f in sorted(out.iterdir()):
        print("  ", f.name)


if __name__ == "__main__":
    main()
