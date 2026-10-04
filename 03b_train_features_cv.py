"""Step 3b - Repeated stratified CV on cached features (fast on CPU).

Methods
  logreg_mean : mean-pooled slice features -> standardise -> L2 logistic regression  (baseline)
  attn_mil    : gated-attention MIL head on slice features                           (main method)

Output per method: work_dir/results/<method>/{oof_predictions.csv, fold_log.csv, attn/*.npz}
Evaluate with:  python 04_evaluate.py --results work/results/<method>
"""
import argparse
import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def safe_auc(y, p):
    try:
        return roc_auc_score(y, p[:, 1]) if p.shape[1] == 2 else roc_auc_score(y, p, multi_class="ovr")
    except ValueError:
        return float("nan")


class FeatMIL(nn.Module):
    def __init__(self, d, n_classes, hidden=128, drop=0.3):
        super().__init__()
        self.proj = nn.Sequential(nn.Dropout(drop), nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(drop))
        self.V, self.U, self.w = nn.Linear(hidden, 64), nn.Linear(hidden, 64), nn.Linear(64, 1)
        self.cls = nn.Linear(hidden, n_classes)

    def forward(self, x):                                   # (B, N, D)
        h = self.proj(x)
        a = torch.softmax(self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))).squeeze(-1), 1)
        return self.cls((a.unsqueeze(-1) * h).sum(1)), a


def sample_bags(X, bag):
    out = []
    for x in X:
        Z = len(x)
        idx = random.sample(range(Z), bag) if Z >= bag else random.choices(range(Z), k=bag)
        out.append(x[sorted(idx)])
    return torch.from_numpy(np.stack(out))


@torch.no_grad()
def mil_predict(model, X, y=None, crit=None):
    model.eval()
    P, A, loss = [], [], 0.0
    for i, x in enumerate(X):
        logits, a = model(torch.from_numpy(x)[None])
        P.append(torch.softmax(logits, 1)[0].numpy()); A.append(a[0].numpy())
        if crit is not None:
            loss += crit(logits, torch.tensor([y[i]])).item()
    return np.array(P), A, loss / max(1, len(X))


def fit_mil(Xtr, ytr, Xva, yva, n_classes, mc, seed):
    seed_all(seed)
    model = FeatMIL(Xtr[0].shape[1], n_classes, mc["hidden"], mc["dropout"])
    counts = np.bincount(ytr, minlength=n_classes).clip(1)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(counts.sum() / (n_classes * counts), dtype=torch.float32))
    opt = torch.optim.AdamW(model.parameters(), lr=mc["lr"], weight_decay=mc["weight_decay"])
    best, state, bad = 1e9, None, 0
    order = np.arange(len(Xtr))
    for ep in range(mc["epochs"]):
        model.train()
        np.random.shuffle(order)
        for s in range(0, len(order), mc["batch_size"]):
            b = order[s:s + mc["batch_size"]]
            x = sample_bags([Xtr[i] for i in b], mc["bag_size"])
            loss = crit(model(x)[0], torch.tensor(ytr[b]))
            opt.zero_grad(); loss.backward(); opt.step()
        _, _, vloss = mil_predict(model, Xva, yva, crit)
        if vloss < best - 1e-4:
            best, state, bad = vloss, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= mc["patience"]:
                break
    model.load_state_dict(state)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--methods", nargs="+", default=["logreg_mean", "attn_mil"], choices=["logreg_mean", "attn_mil"])
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    tc, mc, work = cfg["train"], cfg["feature_mil"], Path(cfg["paths"]["work_dir"])
    torch.set_num_threads(cfg["features"].get("num_threads", 4))

    df = pd.read_csv(work / cfg["paths"].get("labels_file", "labels.csv"))
    fdir = work / "features" / cfg["features"]["backbone"]
    df = df[df.patient_id.map(lambda p: (fdir / f"{p}.npy").exists())].reset_index(drop=True)
    feats = {p: np.load(fdir / f"{p}.npy").astype(np.float32) for p in df.patient_id}
    y_all = df.label.values.astype(int)
    n_classes = int(y_all.max()) + 1
    n_splits = int(min(tc["n_splits"], np.bincount(y_all).min()))
    print(f"{len(df)} patients, {n_classes} classes, {n_splits}-fold x {tc['n_repeats']} repeats")

    for method in args.methods:
        res = work / cfg["paths"].get("results_dir", "results") / method
        (res / "attn").mkdir(parents=True, exist_ok=True)
        rows, logs = [], []
        for rep in range(tc["n_repeats"]):
            skf = StratifiedKFold(n_splits, shuffle=True, random_state=tc["seed"] + rep)
            for fold, (a, b) in enumerate(skf.split(df, y_all)):
                seed = tc["seed"] + 100 * rep + fold
                tr_ids, te_ids = df.patient_id.values[a], df.patient_id.values[b]
                ytr, yte = y_all[a], y_all[b]
                attn = None

                if method == "logreg_mean":
                    Xtr = np.stack([feats[p].mean(0) for p in tr_ids]); Xte = np.stack([feats[p].mean(0) for p in te_ids])
                    clf = make_pipeline(StandardScaler(), LogisticRegression(C=mc["logreg_C"], class_weight="balanced", max_iter=3000))
                    clf.fit(Xtr, ytr)
                    P = clf.predict_proba(Xte)
                else:
                    try:
                        i_tr, i_va = train_test_split(np.arange(len(a)), test_size=tc["val_fraction"], stratify=ytr, random_state=seed)
                    except ValueError:
                        i_tr, i_va = train_test_split(np.arange(len(a)), test_size=tc["val_fraction"], random_state=seed)
                    allf = np.concatenate([feats[tr_ids[i]] for i in i_tr])
                    mu, sd = allf.mean(0), allf.std(0) + 1e-6            # standardise with training stats only
                    norm = lambda ids: [((feats[p] - mu) / sd).astype(np.float32) for p in ids]
                    model = fit_mil(norm(tr_ids[i_tr]), ytr[i_tr], norm(tr_ids[i_va]), ytr[i_va], n_classes, mc, seed)
                    P, attn, _ = mil_predict(model, norm(te_ids))

                for k, pid in enumerate(te_ids):
                    rows.append(dict(patient_id=pid, label=int(yte[k]), repeat=rep, fold=fold,
                                     **{f"p{c}": float(P[k][c]) for c in range(n_classes)}))
                    if attn is not None:
                        np.savez(res / "attn" / f"r{rep}_f{fold}_{pid}.npz", idx=np.arange(len(attn[k])), w=attn[k])
                logs.append(dict(repeat=rep, fold=fold, test_auc=safe_auc(yte, P)))
                print(f"[{method} rep {rep} fold {fold}] AUC={logs[-1]['test_auc']:.3f}")
                pd.DataFrame(rows).to_csv(res / "oof_predictions.csv", index=False)
                pd.DataFrame(logs).to_csv(res / "fold_log.csv", index=False)


if __name__ == "__main__":
    main()
