#!/usr/bin/env python3
"""
check_vqc_invariance_direct.py
================================
Direct test of the unitary-invariance claim: does M1c's fidelity score
actually change if the VQC's trained q_weights are swapped back out for
the ORIGINAL random initialization?

This is the falsifiable consequence of the argument, unlike parameter
drift (which can look large purely from Adam's random-walk behavior on
a near-zero-gradient parameter -- see check_vqc_training.py's caveat).

If the argument holds: M1c's scores/F1 should be IDENTICAL (up to
floating point) whether q_weights is trained or freshly re-initialized,
because fidelity between full VQC output states is provably invariant
to any unitary applied identically to every sample.

Usage
-----
    python3 check_vqc_invariance_direct.py <split_dir> <training_script.py> <data_dir> <prefix>

Example
-------
    python3 check_vqc_invariance_direct.py \\
        "results/run0/split7" \\
        "Group1_cade_comp_fidelity_5q.py" \\
        "/home/snowm001/projects/OOD/Data/cade_lamda" \\
        "drebin_new"
"""

import os
import sys
import json
import importlib.util

import numpy as np
import torch


def load_training_module(script_path):
    spec = importlib.util.spec_from_file_location("training_script", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    if len(sys.argv) != 5:
        print(__doc__)
        sys.exit(1)

    split_dir, script_path, data_dir, prefix = sys.argv[1:5]

    summary = json.load(open(os.path.join(split_dir, "summary.json")))
    input_dim  = summary["n_features"]
    seed       = summary["seed"]
    split_id   = summary["split_id"]
    ood_family = summary.get("ood_family", "unknown")
    reported_m1c_f1 = summary["M1c_Fidelity_DensityPrototype"]["F1"]

    print("=" * 70)
    print(f"split_dir={split_dir}  ood_family={ood_family}  seed={seed}")
    print(f"Reported M1c F1 (from summary.json): {reported_m1c_f1:.4f}")
    print("=" * 70)

    mod = load_training_module(script_path)

    if not hasattr(mod, "AmplitudeQuantumEncoder"):
        print("This script has no VQC (AmplitudeQuantumEncoder not found) -- "
              "this check does not apply.")
        sys.exit(0)

    # ---- Load data exactly as the training script does ----
    npz_path = os.path.join(os.path.expanduser(data_dir), f"{prefix}_{split_id}.npz")
    d = np.load(npz_path, allow_pickle=True)
    _xt = d["X_train"]
    X_train = _xt.item().toarray().astype(np.float32) if _xt.shape == () else _xt.astype(np.float32)
    y_train = d["y_train"].astype(np.int64)
    _xte = d["X_test"]
    X_test  = _xte.item().toarray().astype(np.float32) if _xte.shape == () else _xte.astype(np.float32)
    y_test  = d["y_test"].astype(np.int64)

    # ---- Build TWO models: trained weights, and fresh/untrained weights ----
    trained_model = mod.AmplitudeQuantumEncoder(input_dim=input_dim, seed=seed)
    state_dict = torch.load(os.path.join(split_dir, "model_weights.pt"), map_location="cpu")
    trained_model.load_state_dict(state_dict)
    trained_model.eval()

    fresh_model = mod.AmplitudeQuantumEncoder(input_dim=input_dim, seed=seed)
    # Give fresh_model the SAME trained compressor/decoder (isolate q_weights
    # as the only difference -- otherwise the compressor itself would differ
    # too, since it's freshly initialized, confounding the comparison)
    fresh_state = {k: v for k, v in state_dict.items()}
    fresh_state["q_weights"] = fresh_model.q_weights.detach().clone()
    fresh_model.load_state_dict(fresh_state)
    fresh_model.eval()

    print("\nq_weights trained vs fresh -- confirming they actually differ:")
    print(f"  ||trained|| = {trained_model.q_weights.norm().item():.6f}")
    print(f"  ||fresh||   = {fresh_model.q_weights.norm().item():.6f}")
    print(f"  ||diff||    = {(trained_model.q_weights - fresh_model.q_weights).norm().item():.6f}")

    # ---- Recompute M1c (fidelity/density-matrix) scoring under BOTH ----
    def compute_m1c_f1(model):
        states_train = model.get_states_numpy(X_train)
        states_test  = model.get_states_numpy(X_test)
        rho_dict = mod.build_density_matrices(states_train, y_train)
        scores = mod.compute_fidelity_scores(states_test, rho_dict)
        # Determine the numeric OOD label: the class present in y_test but
        # absent from y_train (the held-out family), matching how run_task
        # derives it.
        known = set(np.unique(y_train).tolist())
        candidates = set(np.unique(y_test).tolist()) - known
        if len(candidates) != 1:
            raise RuntimeError(f"Expected exactly one OOD label, found: {candidates}")
        ood_label = candidates.pop()
        res = mod.evaluate_ood(scores, y_test, ood_label)
        return res["best_f1"], scores

    trained_f1, trained_scores = compute_m1c_f1(trained_model)
    fresh_f1, fresh_scores     = compute_m1c_f1(fresh_model)

    corr = np.corrcoef(trained_scores, fresh_scores)[0, 1]
    max_abs_score_diff = np.max(np.abs(trained_scores - fresh_scores))
    mean_abs_score_diff = np.mean(np.abs(trained_scores - fresh_scores))

    print("\n" + "=" * 70)
    print("M1c (Fidelity) results: TRAINED q_weights vs FRESH/RANDOM q_weights")
    print("=" * 70)
    print(f"  Reported F1 (summary.json):     {reported_m1c_f1:.4f}")
    print(f"  Recomputed F1, trained weights: {trained_f1:.4f}")
    print(f"  Recomputed F1, fresh weights:   {fresh_f1:.4f}")
    print(f"  |F1 difference|:                {abs(trained_f1 - fresh_f1):.6f}")
    print(f"  Per-sample score correlation:   {corr:.6f}  (1.0 = identical ranking)")
    print(f"  Max |score diff| per sample:    {max_abs_score_diff:.6f}")
    print(f"  Mean |score diff| per sample:   {mean_abs_score_diff:.6f}")

    print("\nInterpretation:")
    print("  F1 difference ~0 AND correlation ~1.0")
    print("    -> CONFIRMS the invariance argument: q_weights' large drift was")
    print("       numerically real but functionally irrelevant to M1c.")
    print("  F1 difference large OR correlation well below 1.0")
    print("    -> argument does NOT hold as cleanly in practice as the math")
    print("       predicts (possibly floating-point effects in the simulator,")
    print("       or a flaw in this reasoning) -- worth investigating further.")


if __name__ == "__main__":
    main()
