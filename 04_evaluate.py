"""Step 4 - Metrics (with patient-level bootstrap CIs), ROC, confusion matrix, attention examples.

Output (work_dir/results): metrics.json, roc.png, confusion_matrix.png, attention_examples.png
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
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, confusion_matrix,
                             roc_auc_score, roc_curve)


def auc(y, p):
    return roc_auc_score(y, p[:, 1]) if p.shape[1] == 2 else roc_auc_score(y, p, multi_class="ovr", average="macro")


def bootstrap_ci(y, p, fn, n=2000, seed=0):
    rng, vals = np.random.default_rng(seed), []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < len(np.unique(y)):
            continue
        vals.append(fn(y[i], p[i]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--results", default=None, help="results folder (default: work_dir/results)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    res = Path(args.results) if args.results else Path(cfg["paths"]["work_dir"]) / cfg["paths"].get("results_dir", "results")
    oof = pd.read_csv(res / "oof_predictions.csv")
    pcols = [c for c in oof.columns if c.startswith("p") and c[1:].isdigit()]
    K = len(pcols)

    # average over CV repeats -> one prediction per patient
    pat = oof.groupby("patient_id").agg(label=("label", "first"), **{c: (c, "mean") for c in pcols}).reset_index()
    y, p = pat.label.values, pat[pcols].values
    pred = p.argmax(1)
    out = {"n_patients": int(len(pat)), "class_counts": {int(k): int(v) for k, v in pd.Series(y).value_counts().items()}}

    out["auc"] = float(auc(y, p)); out["auc_ci95"] = bootstrap_ci(y, p, auc)
    rep_aucs = [auc(g.label.values, g[pcols].values) for _, g in oof.groupby("repeat")]
    out["auc_per_repeat_mean_sd"] = [float(np.mean(rep_aucs)), float(np.std(rep_aucs))]
    out["balanced_acc"] = float(balanced_accuracy_score(y, pred))
    out["balanced_acc_ci95"] = bootstrap_ci(y, p, lambda a, b: balanced_accuracy_score(a, b.argmax(1)))

    if K == 2:
        out["average_precision"] = float(average_precision_score(y, p[:, 1]))
        fpr, tpr, thr = roc_curve(y, p[:, 1])
        j = int(np.argmax(tpr - fpr))        # Youden threshold, chosen on the same OOF data -> optimistic
        yh = (p[:, 1] >= thr[j]).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, yh).ravel()
        out["youden_threshold"] = float(thr[j])
        out["sensitivity"], out["specificity"] = float(tp / (tp + fn)), float(tn / (tn + fp))
        out["note"] = "Youden threshold selected on the evaluation predictions; report as exploratory."
        plt.figure(figsize=(4.5, 4.5))
        plt.plot(fpr, tpr, lw=2, label=f"AUC={out['auc']:.3f} [{out['auc_ci95'][0]:.2f}-{out['auc_ci95'][1]:.2f}]")
        plt.plot([0, 1], [0, 1], "k--", lw=1)
        plt.xlabel("1 - Specificity"); plt.ylabel("Sensitivity"); plt.legend(loc="lower right"); plt.tight_layout()
        plt.savefig(res / "roc.png", dpi=200); plt.close()
        cm = confusion_matrix(y, yh)
    else:
        cm = confusion_matrix(y, pred)

    plt.figure(figsize=(4, 4)); plt.imshow(cm, cmap="Blues")
    for (i, k), v in np.ndenumerate(cm):
        plt.text(k, i, v, ha="center", va="center")
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
    plt.savefig(res / "confusion_matrix.png", dpi=200); plt.close()

    # attention examples: most confident correct predictions
    conf = pat.assign(pred=pred, conf=p.max(1))
    conf = conf[conf.pred == conf.label].sort_values("conf", ascending=False).head(6)
    cache = Path(cfg["paths"]["work_dir"]) / "cache"
    fig, axes = plt.subplots(len(conf), 3, figsize=(7, 2.4 * max(1, len(conf))), squeeze=False)
    for r, (_, row) in enumerate(conf.iterrows()):
        files = sorted((res / "attn").glob(f"r0_f*_{row.patient_id}.npz"))
        if not files:
            continue
        d = np.load(files[0]); vol = np.load(cache / f"{row.patient_id}.npy", mmap_mode="r")
        for c, k in enumerate(np.argsort(-d["w"])[:3]):
            ax = axes[r, c]; ax.imshow(vol[d["idx"][k]], cmap="gray"); ax.axis("off")
            ax.set_title(f"{row.patient_id[-4:]} y={row.label} a={d['w'][k]:.2f}", fontsize=7)
    plt.tight_layout(); plt.savefig(res / "attention_examples.png", dpi=200); plt.close()

    json.dump(out, open(res / "metrics.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
