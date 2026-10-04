"""Step 6 - Is the imaging AUC real, or a shortcut? (run after 03a and 02_labels.py)

Checks (all use the same patients, same repeated stratified CV, same logistic-regression head):
  A. Label-permutation test    : AUC when labels are shuffled should be ~0.5 -> p-value for the real AUC
  B. Shortcut baselines        : can the label be predicted WITHOUT looking at the image?
                                   - site        (TCGA tissue source site = 2nd field of the patient ID)
                                   - acquisition (slice thickness, #slices, spacing, scan extent, kernel)
                                   - age
  C. Site-grouped CV           : imaging model where whole sites are held out of training
Output: work_dir/results/sanity_checks.json and a printed summary.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_clf(C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=3000))


def cv_auc(X, y, C, seeds, n_splits=5):
    """Mean over repeats of the pooled out-of-fold AUC."""
    aucs = []
    for s in seeds:
        oof = np.zeros(len(y))
        for a, b in StratifiedKFold(n_splits, shuffle=True, random_state=s).split(X, y):
            oof[b] = make_clf(C).fit(X[a], y[a]).predict_proba(X[b])[:, 1]
        aucs.append(roc_auc_score(y, oof))
    return float(np.mean(aucs))


def group_cv_auc(X, y, groups, C):
    n = min(5, len(np.unique(groups)))
    oof, fold_aucs = np.full(len(y), np.nan), []
    for a, b in GroupKFold(n).split(X, y, groups):
        if len(np.unique(y[a])) < 2:
            continue
        oof[b] = make_clf(C).fit(X[a], y[a]).predict_proba(X[b])[:, 1]
        if len(np.unique(y[b])) == 2:
            fold_aucs.append(roc_auc_score(y[b], oof[b]))
    ok = ~np.isnan(oof)
    return float(roc_auc_score(y[ok], oof[ok])), (float(np.mean(fold_aucs)) if fold_aucs else float("nan")), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--n_perm", type=int, default=100)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    work, tc = Path(cfg["paths"]["work_dir"]), cfg["train"]
    C = cfg["feature_mil"]["logreg_C"]
    fdir = work / "features" / cfg["features"]["backbone"]

    lab = pd.read_csv(work / cfg["paths"].get("labels_file", "labels.csv"))
    lab = lab[lab.patient_id.map(lambda p: (fdir / f"{p}.npy").exists())].reset_index(drop=True)
    y = lab.label.values.astype(int)
    if y.max() > 1:
        raise SystemExit("These checks are written for binary labels.")
    n_splits = int(min(5, np.bincount(y).min()))
    seeds = [tc["seed"] + r for r in range(tc["n_repeats"])]

    # ---- metadata tables -------------------------------------------------
    used = pd.read_csv(work / "series_used.csv")[["patient_id", "thickness", "n_slices", "kernel"]]
    zq = pd.read_csv(work / "series_zqc.csv")[["patient_id", "median_spacing_mm", "extent_mm"]]
    meta = lab[["patient_id", "label"]].merge(used, on="patient_id", how="left").merge(zq, on="patient_id", how="left")
    meta["site"] = meta.patient_id.str.split("-").str[1]

    print("=== Patients per tissue-source site ===")
    st = meta.groupby("site").label.agg(n="count", positives="sum")
    st["positive_rate"] = (st.positives / st.n).round(2)
    print(st.sort_values("n", ascending=False).to_string(), "\n")

    num_cols = ["thickness", "n_slices", "median_spacing_mm", "extent_mm"]
    acq = meta[num_cols].apply(pd.to_numeric, errors="coerce")
    acq = acq.fillna(acq.median())
    k = meta.kernel.fillna("NA").astype(str)
    k = k.where(k.map(k.value_counts()) >= 5, "other")                # rare kernels pooled
    acq = pd.concat([acq, pd.get_dummies(k, prefix="kernel").astype(float)], axis=1)
    X_acq = acq.values.astype(float)
    X_site = pd.get_dummies(meta.site).values.astype(float)

    X_img = np.stack([np.load(fdir / f"{p}.npy").astype(np.float32).mean(0) for p in lab.patient_id])

    res = {"n_patients": int(len(y)), "n_positive": int(y.sum())}

    # ---- observed imaging AUC (same procedure as everything below) -------
    obs = cv_auc(X_img, y, C, seeds, n_splits)
    res["imaging_logreg_auc"] = obs

    # ---- A. permutation test ----------------------------------------------
    rng, null = np.random.default_rng(0), []
    for i in range(args.n_perm):
        null.append(cv_auc(X_img, rng.permutation(y), C, seeds, n_splits))
        if (i + 1) % 20 == 0:
            print(f"  permutation {i + 1}/{args.n_perm}")
    null = np.array(null)
    res["permutation"] = {"n_perm": args.n_perm, "null_mean": float(null.mean()),
                          "null_95th_percentile": float(np.percentile(null, 95)),
                          "p_value": float((1 + np.sum(null >= obs)) / (1 + args.n_perm))}

    # ---- B. shortcut baselines --------------------------------------------
    res["baseline_site_only_auc"] = cv_auc(X_site, y, C, seeds, n_splits)
    res["baseline_acquisition_only_auc"] = cv_auc(X_acq, y, C, seeds, n_splits)
    try:                                                              # age from the clinical table
        cl = pd.read_csv(cfg["paths"]["clinical_file"], sep=None, engine="python", comment="#")
        pid = [c for c in cl.columns if c.lower() in ("patient_id", "patient id", "bcr_patient_barcode")][0]
        agec = [c for c in cl.columns if c.lower() in ("age", "diagnosis age", "age_at_initial_pathologic_diagnosis")][0]
        cl["patient_id"] = cl[pid].astype(str).str[:12]
        age = meta[["patient_id"]].merge(cl[["patient_id", agec]].drop_duplicates("patient_id"), on="patient_id", how="left")[agec]
        age = pd.to_numeric(age, errors="coerce")
        res["baseline_age_only_auc"] = cv_auc(age.fillna(age.median()).values.reshape(-1, 1), y, C, seeds, n_splits)
    except Exception as e:
        res["baseline_age_only_auc"] = None
        print(f"[note] age baseline skipped: {e}")

    # ---- C. site-grouped CV -------------------------------------------------
    pooled, per_fold, n_groups = group_cv_auc(X_img, y, meta.site.values, C)
    res["imaging_site_grouped_cv"] = {"pooled_auc": pooled, "mean_fold_auc": per_fold, "n_site_folds": n_groups}

    (work / cfg["paths"].get("results_dir", "results")).mkdir(exist_ok=True)
    json.dump(res, open(work / cfg["paths"].get("results_dir", "results") / "sanity_checks.json", "w"), indent=2)

    p = res["permutation"]
    print("\n=== SUMMARY ===")
    print(f"Imaging (mean-pooled features + logreg)      AUC = {obs:.3f}")
    print(f"A. Shuffled labels                            AUC = {p['null_mean']:.3f} (95th pct {p['null_95th_percentile']:.3f}),"
          f" p = {p['p_value']:.3f}")
    print(f"B. Site only                                  AUC = {res['baseline_site_only_auc']:.3f}")
    print(f"B. Acquisition only (thickness, #slices, ...) AUC = {res['baseline_acquisition_only_auc']:.3f}")
    if res["baseline_age_only_auc"] is not None:
        print(f"B. Age only                                   AUC = {res['baseline_age_only_auc']:.3f}")
    g = res["imaging_site_grouped_cv"]
    print(f"C. Imaging, whole sites held out (n={g['n_site_folds']} folds) pooled AUC = {g['pooled_auc']:.3f},"
          f" mean fold AUC = {g['mean_fold_auc']:.3f}")
    print("\nHow to read: A should be ~0.5. If B (site or acquisition) is close to the imaging AUC, or C drops toward 0.5,"
          "\nthe imaging result may reflect site/scanner rather than disease. If B is near 0.5 and C stays well above it,"
          "\nthe imaging signal is more credible.")


if __name__ == "__main__":
    main()
