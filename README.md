Optional first check (fake data, ~1 min, no DICOM needed):  python smoke_test.py

# TCGA-OV CT severity pipeline

Run in order (edit `config.yaml` first):

    pip install -r requirements.txt
    python 01_preprocess.py      # scan DICOM, pick 1 series/patient, cache volumes, QC montages
    python 02_labels.py          # build labels from clinical table; prints class counts
    # --- CPU route (recommended without a GPU) ---
    python 03a_extract_features.py   # frozen ResNet features, run once
    python 03b_train_features_cv.py  # logreg baseline + attention-MIL on cached features
    python 04_evaluate.py --results work/results/attn_mil
    python 04_evaluate.py --results work/results/logreg_mean
    python 05_compare_methods.py     # thesis table + paired bootstrap of AUC difference
    python 06_sanity_checks.py       # permutation test, site/acquisition/age shortcut baselines, site-held-out CV
    python 07_incremental_value.py   # does imaging add information beyond age? (nested late fusion)

    # --- GPU route (Colab/Kaggle) ---
    python 03_train_cv.py            # end-to-end fine-tuned attention-MIL
    python 04_evaluate.py

Before training, check:
1. `work/series_selected.csv` and `work/qc/*.png` (right series? sensible crop?)
2. class counts printed by `02_labels.py`

Design notes
- Split is always at patient level (no slice leakage).
- Out-of-fold predictions from every repeat are averaged per patient for final metrics.
- No tumour masks are required; attention weights indicate which slices drove the prediction.
