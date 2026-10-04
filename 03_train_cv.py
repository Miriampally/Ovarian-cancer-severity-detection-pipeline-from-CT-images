"""Step 3 - Patient-level repeated stratified CV with an attention-based multiple-instance model.

Each patient = a bag of 2.5D CT slices (3 neighbouring slices -> 3 channels).
No tumour masks are needed: a pretrained ResNet encodes each slice, gated attention
pools slices into one patient vector, and a linear head predicts severity.

Output (work_dir/results):
  oof_predictions.csv   out-of-fold probabilities (patient, repeat, fold, p0..pk)
  attn/*.npz            slice indices + attention weights per test patient
  fold_log.csv          per-fold training summary
"""
import argparse
import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class BagDataset(Dataset):
    def __init__(self, df, cache, bag_size, train):
        self.pids, self.labels = df.patient_id.tolist(), df.label.astype(int).tolist()
        self.cache, self.bag, self.train = Path(cache), bag_size, train
        self.geo = v2.Compose([v2.RandomHorizontalFlip(),
                               v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1))])

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, i):
        vol = np.load(self.cache / f"{self.pids[i]}.npy", mmap_mode="r")
        Z = vol.shape[0]
        if self.train:
            idx = sorted(random.sample(range(Z), self.bag) if Z >= self.bag else random.choices(range(Z), k=self.bag))
            idx = np.array(idx)
        else:
            idx = np.unique(np.linspace(0, Z - 1, self.bag).round().astype(int))
        # 2.5D: previous / current / next slice as 3 channels
        x = np.stack([vol[np.clip(idx - 1, 0, Z - 1)], vol[idx], vol[np.clip(idx + 1, 0, Z - 1)]], 1)
        x = torch.from_numpy(x.astype(np.float32) / 255.0)                  # (N, 3, H, W)
        if self.train:
            x = self.geo(x)                                                 # same params for the whole bag
            x = (x * random.uniform(0.9, 1.1) + random.uniform(-0.05, 0.05)).clamp(0, 1)
        return x, self.labels[i], self.pids[i], torch.from_numpy(idx)


class AttnMIL(nn.Module):
    def __init__(self, n_classes, backbone="resnet18", pretrained=True, freeze=False, dropout=0.25):
        super().__init__()
        m = getattr(torchvision.models, backbone)(weights="DEFAULT" if pretrained else None)
        feat = m.fc.in_features
        m.fc = nn.Identity()
        self.enc = m
        if freeze:
            for p in self.enc.parameters():
                p.requires_grad = False
        self.V, self.U, self.w = nn.Linear(feat, 128), nn.Linear(feat, 128), nn.Linear(128, 1)
        self.cls = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat, n_classes))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):                                   # x: (B, N, 3, H, W)
        B, N = x.shape[:2]
        h = self.enc((x.flatten(0, 1) - self.mean) / self.std).view(B, N, -1)
        a = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))).squeeze(-1)
        a = torch.softmax(a.float(), 1)
        z = (a.unsqueeze(-1).to(h.dtype) * h).sum(1)
        return self.cls(z), a


def set_train(model):
    model.train()
    for m in model.enc.modules():                           # frozen BN stats: more stable with tiny batches
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


@torch.no_grad()
def predict(model, loader, dev, crit=None):
    model.eval()
    P, Y, ids, att, loss_sum = [], [], [], {}, 0.0
    for x, y, pid, idx in loader:
        logits, a = model(x.to(dev))
        if crit is not None:
            loss_sum += crit(logits.float(), y.to(dev)).item() * len(y)
        P.append(torch.softmax(logits.float(), 1).cpu().numpy()); Y += y.tolist(); ids += list(pid)
        att[pid[0]] = (idx[0].numpy(), a[0].cpu().numpy())
    return ids, np.concatenate(P), np.array(Y), loss_sum / max(1, len(Y)), att


def safe_auc(y, p):
    try:
        return roc_auc_score(y, p[:, 1]) if p.shape[1] == 2 else roc_auc_score(y, p, multi_class="ovr")
    except ValueError:
        return float("nan")


def run_fold(cfg, tr, te, n_classes, dev, cache, seed):
    tc = cfg["train"]
    seed_all(seed)
    try:
        tr, va = train_test_split(tr, test_size=tc["val_fraction"], stratify=tr.label, random_state=seed)
    except ValueError:
        tr, va = train_test_split(tr, test_size=tc["val_fraction"], random_state=seed)

    mk = lambda df, train, bag, bs: DataLoader(BagDataset(df, cache, bag, train), batch_size=bs, shuffle=train,
                                               num_workers=tc["num_workers"], pin_memory=dev.type == "cuda",
                                               persistent_workers=tc["num_workers"] > 0)
    ltr, lva, lte = (mk(tr, True, tc["bag_size"], tc["batch_size"]), mk(va, False, tc["eval_bag_size"], 1),
                     mk(te, False, tc["eval_bag_size"], 1))

    model = AttnMIL(n_classes, tc["backbone"], tc["pretrained"], tc["freeze_backbone"]).to(dev)
    counts = np.bincount(tr.label.astype(int), minlength=n_classes).clip(1)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(counts.sum() / (n_classes * counts), dtype=torch.float32, device=dev))
    head = [p for n, p in model.named_parameters() if not n.startswith("enc.") and p.requires_grad]
    enc = [p for p in model.enc.parameters() if p.requires_grad]
    groups = [{"params": head, "lr": tc["lr_head"]}] + ([{"params": enc, "lr": tc["lr_backbone"]}] if enc else [])
    opt = torch.optim.AdamW(groups, weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tc["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")

    best, best_state, bad, best_ep = 1e9, None, 0, 0
    for ep in range(tc["epochs"]):
        set_train(model)
        for x, y, _, _ in ltr:
            x, y = x.to(dev), y.to(dev)
            with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
                logits, _ = model(x)
            loss = crit(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sched.step()
        _, pv, yv, vloss, _ = predict(model, lva, dev, crit)
        if vloss < best - 1e-4:
            best, best_state, bad, best_ep = vloss, copy.deepcopy(model.state_dict()), 0, ep
        else:
            bad += 1
            if bad >= tc["patience"]:
                break
    model.load_state_dict(best_state)
    ids, P, Y, _, att = predict(model, lte, dev)
    return ids, P, Y, att, dict(best_epoch=best_ep, val_loss=best, test_auc=safe_auc(Y, P))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    tc, work = cfg["train"], Path(cfg["paths"]["work_dir"])
    res = work / "results"; (res / "attn").mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", dev)

    df = pd.read_csv(work / "labels.csv")
    n_classes = int(df.label.max()) + 1
    n_splits = int(min(tc["n_splits"], df.label.value_counts().min()))
    print(f"{len(df)} patients, {n_classes} classes, {n_splits}-fold x {tc['n_repeats']} repeats")

    rows, logs = [], []
    for rep in range(tc["n_repeats"]):
        skf = StratifiedKFold(n_splits, shuffle=True, random_state=tc["seed"] + rep)
        for fold, (a, b) in enumerate(skf.split(df, df.label)):
            ids, P, Y, att, info = run_fold(cfg, df.iloc[a], df.iloc[b], n_classes, dev,
                                            work / "cache", tc["seed"] + 100 * rep + fold)
            for pid, p, y in zip(ids, P, Y):
                rows.append(dict(patient_id=pid, label=int(y), repeat=rep, fold=fold,
                                 **{f"p{k}": float(p[k]) for k in range(n_classes)}))
                np.savez(res / "attn" / f"r{rep}_f{fold}_{pid}.npz", idx=att[pid][0], w=att[pid][1])
            logs.append(dict(repeat=rep, fold=fold, **info))
            print(f"[rep {rep} fold {fold}] {info}")
            pd.DataFrame(rows).to_csv(res / "oof_predictions.csv", index=False)
            pd.DataFrame(logs).to_csv(res / "fold_log.csv", index=False)


if __name__ == "__main__":
    main()
