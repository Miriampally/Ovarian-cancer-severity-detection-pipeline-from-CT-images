"""Step 3a - Extract frozen ImageNet-pretrained CNN features for every slice (run ONCE, CPU is fine).

Input : work_dir/cache/<patient>.npy         (Z, H, W) uint8
Output: work_dir/features/<backbone>/<patient>.npy   (Z, D) float16, 2.5D input (prev/current/next slice)
Resumable: patients already done are skipped.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
import yaml
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    fc, work = cfg["features"], Path(cfg["paths"]["work_dir"])
    torch.set_num_threads(fc.get("num_threads", 4))

    model = getattr(torchvision.models, fc["backbone"])(weights="DEFAULT")
    model.fc = nn.Identity()
    model.eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    out_dir = work / "features" / fc["backbone"]
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted((work / "cache").glob("*.npy"))
    for f in tqdm(files, desc="Extracting"):
        out = out_dir / f.name
        if out.exists():
            continue
        vol = np.load(f)
        Z, feats = vol.shape[0], []
        with torch.inference_mode():
            for s in range(0, Z, fc["batch_size"]):
                i = np.arange(s, min(s + fc["batch_size"], Z))
                x = np.stack([vol[np.clip(i - 1, 0, Z - 1)], vol[i], vol[np.clip(i + 1, 0, Z - 1)]], 1)
                x = (torch.from_numpy(x.astype(np.float32) / 255.0) - mean) / std
                feats.append(model(x).numpy())
        np.save(out, np.concatenate(feats).astype(np.float16))
    print(f"Features saved in {out_dir}")


if __name__ == "__main__":
    main()
