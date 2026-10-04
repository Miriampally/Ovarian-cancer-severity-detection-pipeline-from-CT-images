"""Run a named variant of the analysis end to end, writing to its OWN labels file and results folder
(your main results are never overwritten). Your config.yaml is the base; the variant only changes what is listed.

  python run_all.py --name geometry               include the 7 geometry-flagged patients
  python run_all.py --name resnet50               ResNet-50 features (slow extraction: run it overnight)
  python run_all.py --name stage --clinical tcia_clinical.txt --steps 02    build stage labels, read the class counts
  python run_all.py --name stage --clinical tcia_clinical.txt               then run the full stage analysis

Options:  --steps 02 03a 03b 04 05      choose steps (default depends on the variant)
          --with_checks                 also run 06 (sanity) and 07 (age + imaging, full and complete-case)
          --dry_run                     print the commands only
Results land in work/<results_dir>/ and a copy of the config used is saved as config_<name>.yaml.
"""
import argparse
import copy
import os
import subprocess
import sys

import yaml

VARIANTS = {
    "geometry": dict(
        patch={"labels": {"exclude_severe_geometry": False},
               "paths": {"labels_file": "labels_geometry_incl.csv", "results_dir": "results_geometry_incl"}},
        steps=["02", "03b", "04", "05"]),
    "resnet50": dict(
        patch={"features": {"backbone": "resnet50"},
               "paths": {"results_dir": "results_resnet50"}},
        steps=["03a", "03b", "04", "05"]),
    "stage": dict(
        patch={"labels": {"mode": "stage_34"},
               "paths": {"labels_file": "labels_stage.csv", "results_dir": "results_stage"}},
        steps=["02", "03b", "04", "05"]),
}
METHODS = ["logreg_mean", "attn_mil"]


def merge(a, b):
    for k, v in b.items():
        a[k] = merge(a.get(k, {}), v) if isinstance(v, dict) else v
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, choices=list(VARIANTS))
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--clinical", default=None, help="clinical file to use (needed for --name stage)")
    ap.add_argument("--steps", nargs="+", default=None, choices=["02", "03a", "03b", "04", "05"])
    ap.add_argument("--with_checks", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    cfg = merge(copy.deepcopy(yaml.safe_load(open(args.config))), copy.deepcopy(VARIANTS[args.name]["patch"]))
    if args.clinical:
        cfg["paths"]["clinical_file"] = args.clinical
    if args.name == "stage" and not args.clinical:
        sys.exit("--name stage needs --clinical <path to the TCIA/GDC clinical file>")

    if "02" in (args.steps or VARIANTS[args.name]["steps"]) and not os.path.exists(cfg["paths"]["clinical_file"]) and not args.dry_run:
        sys.exit(f"Clinical file not found: {cfg['paths']['clinical_file']}\n"
                 "Create it first (python fetch_gdc_clinical.py) or pass the right path with --clinical.")

    cfgfile = f"config_{args.name}.yaml"
    yaml.safe_dump(cfg, open(cfgfile, "w"), sort_keys=False)
    work, rdir = cfg["paths"]["work_dir"], cfg["paths"]["results_dir"]
    steps = args.steps or VARIANTS[args.name]["steps"]

    cmds = []
    if "02" in steps:
        cmds.append(["02_labels.py", "--config", cfgfile])
    if "03a" in steps:
        cmds.append(["03a_extract_features.py", "--config", cfgfile])
    if "03b" in steps:
        cmds.append(["03b_train_features_cv.py", "--config", cfgfile])
    if "04" in steps:
        cmds += [["04_evaluate.py", "--config", cfgfile, "--results", f"{work}/{rdir}/{m}"] for m in METHODS]
    if "05" in steps:
        cmds.append(["05_compare_methods.py", "--config", cfgfile])
    if args.with_checks:
        cmds += [["06_sanity_checks.py", "--config", cfgfile],
                 ["07_incremental_value.py", "--config", cfgfile],
                 ["07_incremental_value.py", "--config", cfgfile, "--complete_case"]]

    print(f"Variant '{args.name}': labels -> {cfg['paths'].get('labels_file', 'labels.csv')}, results -> {work}/{rdir}")
    for c in cmds:
        print("\n$ python", " ".join(c))
        if not args.dry_run:
            subprocess.run([sys.executable] + c, check=True)
    print("\nDone." if not args.dry_run else "\n(dry run - nothing executed)")


if __name__ == "__main__":
    main()
