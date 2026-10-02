#!/usr/bin/env python3
"""
filter_lamda_to_4class.py
===========================
Filters the existing 8-family pooled LAMDA dataset (lamda_ood_like_monthwise/)
down to 4 families -- droidkungfu, plankton, smsreg, airpush -- relabeling
them 0-3. Output has the same schema as the 8-family version, so
build_cade_splits_monthwise.py and apply_cade_variance_threshold.py run
UNCHANGED against it afterward.

Usage:
    python3 filter_lamda_to_4class.py \
        --input-dir  lamda_ood_like_monthwise \
        --output-dir lamda_ood_like_monthwise_4class \
        --families droidkungfu plankton smsreg airpush
"""

import os
import json
import argparse

import numpy as np
from scipy.sparse import load_npz, save_npz


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--families", nargs="+", required=True,
                         help="Family names to KEEP, in the desired new label order")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.input_dir, "family_to_label.json")) as f:
        old_family_to_label = json.load(f)

    for fam in args.families:
        if fam not in old_family_to_label:
            raise ValueError(f"Family '{fam}' not found in {args.input_dir}. "
                              f"Available: {list(old_family_to_label.keys())}")

    old_labels_to_keep = [old_family_to_label[f] for f in args.families]
    new_family_to_label = {fam: i for i, fam in enumerate(args.families)}
    old_to_new_label = {old_l: new_family_to_label[fam]
                         for fam, old_l in zip(args.families, old_labels_to_keep)}

    print(f"Keeping families: {args.families}")
    print(f"Old labels: {old_labels_to_keep}  ->  New labels: {list(range(len(args.families)))}")

    for split in ["train", "test"]:
        X = load_npz(os.path.join(args.input_dir, f"X_{split}.npz"))
        meta = np.load(os.path.join(args.input_dir, f"meta_{split}.npz"), allow_pickle=True)

        y_family_old = meta["y_family"]
        keep_mask = np.isin(y_family_old, old_labels_to_keep)

        X_new = X[keep_mask]
        y_family_new = np.array([old_to_new_label[l] for l in y_family_old[keep_mask]],
                                 dtype=np.int64)

        new_meta = {"y_family": y_family_new}
        for key in meta.files:
            if key == "y_family":
                continue
            new_meta[key] = meta[key][keep_mask]

        save_npz(os.path.join(args.output_dir, f"X_{split}.npz"), X_new)
        np.savez_compressed(os.path.join(args.output_dir, f"meta_{split}.npz"), **new_meta)

        print(f"  {split}: {X.shape[0]} -> {X_new.shape[0]} samples "
              f"(kept {100*X_new.shape[0]/X.shape[0]:.1f}%), features unchanged ({X_new.shape[1]})")

        # Per-family counts, for sanity
        for fam in args.families:
            nl = new_family_to_label[fam]
            print(f"    {fam:<14} ({split}): {int((y_family_new == nl).sum())} samples")

    with open(os.path.join(args.output_dir, "family_to_label.json"), "w") as f:
        json.dump(new_family_to_label, f, indent=2)
    with open(os.path.join(args.output_dir, "label_to_family.json"), "w") as f:
        json.dump({str(v): k for k, v in new_family_to_label.items()}, f, indent=2)

    print(f"\nDone. Output: {args.output_dir}")
    print(f"New family_to_label.json: {new_family_to_label}")


if __name__ == "__main__":
    main()
