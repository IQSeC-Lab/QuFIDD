#!/usr/bin/env python3
"""
check_vqc_training.py
======================
Checks whether the VQC's 300 quantum parameters actually moved during
training, or whether they stayed at (near) their random initialization --
which is what the unitary-invariance argument predicts for a purely
fidelity/cosine-based loss (since such a loss is provably constant with
respect to any unitary applied identically to every sample, so its
gradient w.r.t. the VQC's parameters is zero everywhere).

How it works
------------
`set_seed(seed)` is called at the very start of the model's __init__,
before the compressor/decoder/q_weights are created. That means we can
reconstruct the EXACT original initial q_weights just by instantiating
a fresh model with the same seed and input_dim -- set_seed()
unconditionally resets the global RNG, so this reproduces the initial
draw bit-for-bit, regardless of anything that happened earlier in the
original training run.

We then load the actual trained model_weights.pt and compare:
  - q_weights (VQC):        should barely move, if the argument holds
  - compressor/decoder:     will move a lot (real gradient, uncontested)
                             -- included as an internal sanity control

Usage
-----
    python3 check_vqc_training.py <split_dir> <path_to_training_script.py>

Example
-------
    python3 check_vqc_training.py \\
        "results/run0/split0" \\
        "Group1_cade_comp_fidelity_5q.py"

The training script must be importable (same folder, or pass its path)
since this reuses the REAL model class rather than reimplementing it --
avoids any risk of a subtly different reconstruction.
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
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    split_dir, script_path = sys.argv[1], sys.argv[2]

    summary_path = os.path.join(split_dir, "summary.json")
    weights_path = os.path.join(split_dir, "model_weights.pt")

    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Missing: {summary_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Missing: {weights_path}")

    summary = json.load(open(summary_path))
    input_dim = summary["n_features"]
    seed = summary["seed"]
    ood_family = summary.get("ood_family", "unknown")

    print("=" * 70)
    print(f"Checking split: {split_dir}")
    print(f"  ood_family = {ood_family}   seed = {seed}   input_dim = {input_dim}")
    print("=" * 70)

    mod = load_training_module(script_path)

    # Find the encoder class regardless of which naming convention this
    # script uses (VQC version: AmplitudeQuantumEncoder; classical-mixer
    # version: ClassicalCapacityMatchedEncoder -- this check is only
    # meaningful for the VQC version, but we detect either for clarity).
    if hasattr(mod, "AmplitudeQuantumEncoder"):
        EncoderClass = mod.AmplitudeQuantumEncoder
        has_vqc = True
    elif hasattr(mod, "ClassicalCapacityMatchedEncoder"):
        EncoderClass = mod.ClassicalCapacityMatchedEncoder
        has_vqc = False
        print("\nNOTE: this script uses ClassicalCapacityMatchedEncoder (no VQC) --")
        print("      the q_weights check below does not apply; skipping that part.\n")
    else:
        raise AttributeError(
            "Could not find AmplitudeQuantumEncoder or "
            "ClassicalCapacityMatchedEncoder in the given script."
        )

    # Reconstruct the EXACT original initial state (set_seed() at the top
    # of __init__ resets the global RNG unconditionally, so this reproduces
    # the initial draw bit-for-bit).
    fresh_model = EncoderClass(input_dim=input_dim, seed=seed)

    # Load the actual trained weights.
    trained_state = torch.load(weights_path, map_location="cpu")

    def compare(name, init_tensor, trained_key):
        if trained_key not in trained_state:
            print(f"  [skip] '{trained_key}' not found in model_weights.pt")
            return
        trained_tensor = trained_state[trained_key]
        if init_tensor.shape != trained_tensor.shape:
            print(f"  [skip] '{trained_key}' shape mismatch: "
                  f"{init_tensor.shape} vs {trained_tensor.shape}")
            return
        diff = (trained_tensor - init_tensor).detach()
        init_norm = init_tensor.norm().item()
        diff_norm = diff.norm().item()
        rel_change = diff_norm / (init_norm + 1e-12)
        max_abs = diff.abs().max().item()
        mean_abs = diff.abs().mean().item()
        print(f"  {name}:")
        print(f"      init  ||.||      = {init_norm:.6f}")
        print(f"      trained ||.||    = {trained_tensor.norm().item():.6f}")
        print(f"      ||diff||         = {diff_norm:.6f}")
        print(f"      relative change  = {rel_change:.6f}   (||diff|| / ||init||)")
        print(f"      max |Δ| per elem = {max_abs:.6f}")
        print(f"      mean|Δ| per elem = {mean_abs:.6f}")

    if has_vqc:
        print("\n--- VQC parameters (q_weights) ---")
        print("Prediction if the unitary-invariance argument holds: relative")
        print("change should be tiny (numerical drift only, not real learning).")
        compare("q_weights", fresh_model.q_weights, "q_weights")

    print("\n--- Compressor (for contrast -- SHOULD move a lot, real gradient) ---")
    for pname, ptensor in fresh_model.compressor.state_dict().items():
        compare(f"compressor.{pname}", ptensor, f"compressor.{pname}")

    print("\n--- Decoder (for contrast -- SHOULD move a lot, real gradient) ---")
    for pname, ptensor in fresh_model.decoder.state_dict().items():
        compare(f"decoder.{pname}", ptensor, f"decoder.{pname}")

    print("\n" + "=" * 70)
    print("Interpretation:")
    print("  q_weights relative change << compressor/decoder relative change")
    print("    -> consistent with the unitary-invariance argument: the VQC")
    print("       received ~zero real gradient signal from this loss.")
    print("  q_weights relative change comparable to compressor/decoder")
    print("    -> argument would be empirically contradicted; worth a closer")
    print("       look (e.g. check M1a specifically, which is the one method")
    print("       that COULD reflect real VQC learning).")
    print("=" * 70)


if __name__ == "__main__":
    main()
