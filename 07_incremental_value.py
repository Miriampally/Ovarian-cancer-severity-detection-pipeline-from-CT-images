"""Step 7 - Does imaging add information beyond age?   (binary labels)

Three models, identical repeated stratified CV (same seeds/splits as 03b):
  age      : logistic regression on age alone                     (clinical baseline)
  imaging  : mean-pooled frozen CNN features -> logistic regression
  fusion   : late fusion. Inside every training fold, imaging scores are generated out-of-fold
             (inner CV) and combined with age by a 2-input logistic regression; the test fold is
             scored with the imaging model fitted on the whole training fold. No test data is used
             anywhere in fitting (nested, leakage-free).

Reports pooled AUC (per repeat mean±sd and from repeat-averaged predictions with patient-level
bootstrap CI) and PAIRED bootstrap differences:  fusion - age,  fusion - imaging.

Output: work_dir/results/incremental_value.json, incremental_value_predictions.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def clf(C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=3000))


def load_age(cfg, ids):
    """Age in years from AGE, falling back to -DAYS_TO_BIRTH/365.25. NaN where unknown."""
    cl = pd.read_csv(cfg["paths"]["clinical_file"], sep=None, engine="python", comment="#")
    pid = [c for c in cl.columns if c.lower() in ("patient_id", "patient id", "bcr_patient_barcode")][0]
    agec = [c for c in cl.columns if c.lower() in ("age", "diagnosis age", "age_at_initial_pathologic_diagnosis")][0]
    cl["patient_id"] = cl[pid].astype(str).str[:12]
    extra = [c for c in cl.columns if c.lower() == "days_to_birth"]
    cols = ["patient_id", agec] + extra
    m = pd.DataFrame({"patient_id": ids}).merge(cl[cols].drop_duplicates("patient_id"), on="patient_id", how="left")
    a = pd.to_numeric(m[agec], errors="coerce")
    for c in extra:
        a = a.fillna(-pd.to_numeric(m[c], errors="coerce") / 365.25)
    print(f"age unknown for {int(a.isna().sum())} of {len(a)} patients")
    return a.values.astype(float)


def fusion_fold(Xtr, atr, ytr, Xte, ate, C, seed):
    k = int(min(5, np.bincount(ytr).min()))
    inner = cross_val_predict(clf(C), Xtr, ytr, cv=StratifiedKFold(k, shuffle=True, random_state=seed),
                              method="decision_function")
    s_te = clf(C).fit(Xtr, ytr).decision_function(Xte)
    z_tr, z_te = np.c_[inner, atr], np.c_[s_te, ate]
    head = clf(1.0).fit(z_tr, ytr)
    return head.predict_proba(z_te)[:, 1], head[-1].coef_[0]


def boot(y, preds, n=2000, seed=0):
    rng, names = np.random.default_rng(seed), list(preds)
    A = {m: [] for m in names}
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        for m in names:
            A[m].append(roc_auc_score(y[i], preds[m][i]))
    return {m: np.array(v) for m, v in A.items()}


def ci(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--complete_case", action="store_true", help="only patients with known age (no imputation)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    work, tc, C = Path(cfg["paths"]["work_dir"]), cfg["train"], cfg["feature_mil"]["logreg_C"]
    fdir = work / "features" / cfg["features"]["backbone"]

    lab = pd.read_csv(work / cfg["paths"].get("labels_file", "labels.csv"))
    lab = lab[lab.patient_id.map(lambda p: (fdir / f"{p}.npy").exists())].reset_index(drop=True)
    age = load_age(cfg, lab.patient_id.tolist())
    tag = ""
    if args.complete_case:
        keep = ~np.isnan(age)
        lab, age, tag = lab[keep].reset_index(drop=True), age[keep], "_complete_case"
        print(f"complete-case analysis: {int(keep.sum())} patients with known age")
    else:
        age = np.where(np.isnan(age), np.nanmedian(age), age)       # median imputation (age baseline is handicapped)
    y = lab.label.values.astype(int)
    if y.max() > 1:
        raise SystemExit("Binary labels only.")
    X = np.stack([np.load(fdir / f"{p}.npy").astype(np.float32).mean(0) for p in lab.patient_id])
    n_splits = int(min(tc["n_splits"], np.bincount(y).min()))

    names = ["age", "imaging", "fusion"]
    per_rep = {m: [] for m in names}
    avg = {m: np.zeros(len(y)) for m in names}
    coefs = []
    for rep in range(tc["n_repeats"]):
        oof = {m: np.zeros(len(y)) for m in names}
        for fold, (a, b) in enumerate(StratifiedKFold(n_splits, shuffle=True, random_state=tc["seed"] + rep).split(X, y)):
            seed = tc["seed"] + 100 * rep + fold
            oof["age"][b] = clf(1.0).fit(age[a, None], y[a]).predict_proba(age[b, None])[:, 1]
            oof["imaging"][b] = clf(C).fit(X[a], y[a]).predict_proba(X[b])[:, 1]
            oof["fusion"][b], w = fusion_fold(X[a], age[a], y[a], X[b], age[b], C, seed)
            coefs.append(w)
        for m in names:
            per_rep[m].append(roc_auc_score(y, oof[m]))
            avg[m] += oof[m] / tc["n_repeats"]
        print(f"repeat {rep}: " + "  ".join(f"{m}={per_rep[m][-1]:.3f}" for m in names))

    B = boot(y, avg, args.n_boot)
    out = {"n_patients": int(len(y)), "n_positive": int(y.sum()), "models": {}}
    for m in names:
        out["models"][m] = {"auc_from_repeat_averaged_predictions": float(roc_auc_score(y, avg[m])),
                            "auc_ci95": ci(B[m]),
                            "auc_per_repeat_mean_sd": [float(np.mean(per_rep[m])), float(np.std(per_rep[m]))]}
    out["paired_differences"] = {}
    for a_, b_ in [("fusion", "age"), ("fusion", "imaging"), ("imaging", "age")]:
        d = B[a_] - B[b_]
        out["paired_differences"][f"{a_} - {b_}"] = {
            "delta_auc": float(roc_auc_score(y, avg[a_]) - roc_auc_score(y, avg[b_])), "ci95": ci(d),
            "bootstrap_share_delta_gt_0": float(np.mean(d > 0))}
    w = np.mean(coefs, axis=0)
    out["fusion_head_mean_standardised_coefficients"] = {"imaging_score": float(w[0]), "age": float(w[1])}

    (work / cfg["paths"].get("results_dir", "results")).mkdir(exist_ok=True)
    json.dump(out, open(work / cfg["paths"].get("results_dir", "results") / f"incremental_value{tag}.json", "w"), indent=2)
    pd.DataFrame({"patient_id": lab.patient_id, "label": y, **{f"p_{m}": avg[m] for m in names}}).to_csv(
        work / cfg["paths"].get("results_dir", "results") / f"incremental_value_predictions{tag}.csv", index=False)

    print(f"\n=== {len(y)} patients ({int(y.sum())} positive) ===")
    for m in names:
        r = out["models"][m]
        print(f"{m:8s} AUC {r['auc_from_repeat_averaged_predictions']:.3f}  [{r['auc_ci95'][0]:.3f}-{r['auc_ci95'][1]:.3f}]"
              f"   per-repeat {r['auc_per_repeat_mean_sd'][0]:.3f}±{r['auc_per_repeat_mean_sd'][1]:.3f}")
    print("\nPaired bootstrap differences in AUC:")
    for k, v in out["paired_differences"].items():
        print(f"  {k:20s} {v['delta_auc']:+.3f}  [95% CI {v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]")
    print("\nFusion head weights (standardised):", {k: round(v, 2) for k, v in out["fusion_head_mean_standardised_coefficients"].items()})
    print("Reading: if 'fusion - age' has a CI above 0, imaging adds information beyond age. "
          "A CI that includes 0 means the data can't show it (common with ~100 patients) - report that honestly.")


if __name__ == "__main__":
    main()
