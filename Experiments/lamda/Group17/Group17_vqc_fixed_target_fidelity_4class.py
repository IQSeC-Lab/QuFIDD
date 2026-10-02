#!/usr/bin/env python3
"""
Group17_vqc_fixed_target_fidelity_4class.py
==============================================
VQC trained with a FIXED-TARGET fidelity loss -- escapes both structural
traps found in the earlier fidelity-based scoring (G1/G6's M1b/M1c),
which we PROVED are mathematically invariant to the VQC's parameters
(unitary-invariance of full-state inner products between two
theta-evolved states), and the reduced-density-matrix alternative
(M1c'), which we showed is biased in the WRONG direction by fidelity's
monotonicity under partial trace.

Key idea: instead of comparing the transformed state to OTHER
theta-evolved states (which cancels, per the invariance proof), compare
it to a FIXED reference that never passes through U(theta) at all:

    L = 1 - |<target_y|psi(theta)>|^2 = 1 - fidelity(psi, target_y)

Since target_y appears on only ONE side of the inner product, U(theta)
does not cancel -- d(fidelity)/d(theta) is generically nonzero. This is
the direct quantum analog of training a network's output toward a fixed
one-hot target, a completely standard classical technique.

Targets: the 4 classes are assigned to 4 of the 32 computational basis
states (|00000>, |00001>, |00010>, |00011>), which are automatically
pairwise ORTHOGONAL. This means routing a sample to fidelity~1 with its
OWN target automatically forces near-zero fidelity with every other
target -- no explicit "push apart" loss term is needed.

Mathematical simplification used throughout: since target_c is the
computational basis state |c>, <target_c|psi> = psi[c] directly (the
c-th amplitude) -- fidelity to a fixed basis-state target is just
|psi[c]|^2, the probability of measuring bitstring c. No explicit
target vectors are constructed; this indexing IS the fidelity.

Architecture:
  raw features (F-dim)
    -> PCA(F->32), FIXED/frozen (identical role to Group6/Group14's PCA)
    -> AmplitudeEmbedding: 32-dim -> 5-qubit state |psi_in>
    -> VQC: 5 qubits, 20 layers, 200 trainable params (SAME circuit as
       Group14 -- keeping this identical isolates the LOSS as the only
       variable between Group14 and Group17)
    -> full state |psi'> = U(theta)|psi_in>  (qml.state(), 32-dim complex
       -- NOT the 15-dim XYZ readout Group14 uses; fidelity needs the
       full state)
    -> decoder: |psi'|^2 (32 real probabilities) -> classical decoder ->
       reconstructs PCA output. RECON_WEIGHT is a toggle (default 1.0),
       built in from the start this time: unlike the earlier
       contrastive+reconstruction designs, THIS loss has no cheap
       "collapse everything to one point" shortcut (4 distinct orthogonal
       targets make that strategy score badly), so reconstruction's role
       here is different -- retaining general input structure beyond the
       4-way classification signal, relevant specifically because
       classification-style losses have no explicit incentive to
       preserve information about a genuinely novel (OOD) class. The
       RECON_WEIGHT=0 ablation directly tests whether that concern
       matters in practice, the same way it did for Group15.

Loss: fixed_target_fidelity_loss() -- ordinary shuffled minibatches, NO
CADE-style left/right contrastive pairing needed (each sample is scored
against its own fixed target directly).

Scoring: MAD-threshold and threshold-free evaluation, same protocol as
every other script, but "distance" = 1 - fidelity-to-own-target instead
of Euclidean distance to a data-derived centroid -- targets are FIXED,
never computed from training data.

Capacity: 200 quantum trainable params (identical to Group14), pairs
with Group15's 215-param CADE-mini and Group14's 200-param M1a-fix for
a three-way comparison at matched budget on the same 4-class LAMDA data.

Data: 4-class LAMDA subset (droidkungfu, plankton, smsreg, airpush).

Usage:
    python3 Group17_vqc_fixed_target_fidelity_4class.py <task_id> \
        --data-dir <path to 4-class cade_lamda_vt> \
        --prefix drebin_new \
        --results-dir results

    python3 Group17_vqc_fixed_target_fidelity_4class.py aggregate --results-dir results
"""

import os
import sys
import json
import time
import functools
import copy
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
from sklearn.decomposition import PCA


# ======================================================================
# TEE OUTPUT
# ======================================================================

class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
    def flush(self):
        for s in self.streams:
            s.flush()


def _tee_to_file(log_filename):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(result_dir, *args, **kwargs):
            os.makedirs(result_dir, exist_ok=True)
            log_path = os.path.join(result_dir, log_filename)
            old_stdout = sys.stdout
            with open(log_path, 'w') as log_file:
                sys.stdout = _Tee(old_stdout, log_file)
                try:
                    return func(result_dir, *args, **kwargs)
                finally:
                    sys.stdout = old_stdout
            print(f'  Aggregate log saved: {log_path}')
        return wrapper
    return decorator


# ======================================================================
# CONSTANTS
# ======================================================================

N_CLASSES       = 4                 # 4-class subset, NOT 8
N_SPLITS        = N_CLASSES         # leave-one-family-out: one split per class
SEEDS           = {0: 42, 1: 123, 2: 456}

PCA_DIM         = 32                # frozen PCA output dim -- same as Group6/Group9/Group14
N_QUBITS        = 5
N_LAYERS        = 20                # SAME circuit as Group14 -- only the loss differs
N_QUANTUM_PARAMS = N_QUBITS * N_LAYERS * 2   # = 200
STATE_DIM       = 2 ** N_QUBITS     # = 32, full state dimension (fidelity needs full state,
                                     # not the 15-dim XYZ readout Group14 uses)

DEC_HIDDEN      = 24                # decoder hidden width (uncapped -- decoder params
                                     # are not part of the 200-param quantum budget,
                                     # same convention as Group6/Group14's decoders)
RECON_WEIGHT    = 1.0               # ablation toggle -- set to 0.0 to test whether
                                     # reconstruction pressure actually matters here,
                                     # same diagnostic built into Group15's investigation

# Training (this loss has no CADE-style left/right contrastive pairing --
# ordinary shuffled minibatches, one fidelity term per sample)
CAE_LR          = 0.001             # raised from 0.0001: Group14 at 0.0001 barely moved
                                     # from its near-identity init (Adam's per-step move
                                     # ~LR caps total angle travel); same value as Group15
CAE_BATCH_SIZE  = 64
CAE_EPOCHS      = 250               # ceiling, not a fixed count -- early stopping below
PATIENCE        = 5                  # epochs with no improvement before stopping
REL_MIN_DELTA   = 1e-4               # relative improvement threshold: must beat
                                     # best_loss by > REL_MIN_DELTA * best_loss
                                     # (same value as Group14, no min-epoch floor)
MAD_THRESHOLD   = 3.5

DEFAULT_DATA_DIR    = os.path.expanduser('~/projects/OOD/Data/cade_lamda_4class')
DEFAULT_RESULTS_DIR = 'results'


# ======================================================================
# REPRODUCIBILITY (verbatim from M0a_CADE_original.py)
# ======================================================================

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ======================================================================
# FROZEN PCA COMPRESSOR (verbatim from Group6/Group9/Group13)
# ======================================================================

class PCACompressor:
    """PCA-based dimensionality reduction: F -> PCA_DIM. Frozen after fit."""
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.pca       = PCA(n_components=PCA_DIM)
        self.fitted    = False

    def fit(self, X_train):
        self.pca.fit(X_train)
        self.fitted = True
        var_ratio = self.pca.explained_variance_ratio_.sum()
        print(f'  PCA fitted: {self.input_dim}->{PCA_DIM}  '
              f'explained_variance={var_ratio:.3f}', flush=True)
        return self

    def transform(self, X):
        z = self.pca.transform(X).astype(np.float32)
        norm = np.linalg.norm(z, axis=1, keepdims=True).clip(min=1e-8)
        return z / norm

    def compress_tensor(self, X_tensor):
        X_np = X_tensor.numpy() if hasattr(X_tensor, 'numpy') else X_tensor
        return torch.FloatTensor(self.transform(X_np))



# ======================================================================
# VQC: 5 qubits, 20 layers, FULL STATE readout (fidelity needs the full
# state, not a partial/expectation-value summary)
# ======================================================================

def make_state_circuit(n_qubits, n_layers):
    """
    Amplitude-embed a normalized PCA_DIM-vector into n_qubits, apply the
    SAME hardware-efficient ansatz Group14 uses (RY+RZ per qubit per
    layer, ring CNOT + alternating long-range CNOT) -- identical circuit,
    only the final readout differs: full state vector instead of
    expectation values, since fidelity to a fixed target needs the full
    32-dim state.
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(amplitudes, weights):
        qml.AmplitudeEmbedding(amplitudes, wires=range(n_qubits),
                               normalize=True, pad_with=0.0)
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.RY(weights[layer, i, 0], wires=i)
                qml.RZ(weights[layer, i, 1], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[n_qubits - 1, 0])
            if layer % 2 == 1:
                for i in range(0, n_qubits - 2, 2):
                    qml.CNOT(wires=[i, i + 2])
        return qml.state()

    return circuit


class Decoder(nn.Module):
    """
    STATE_DIM -> DEC_HIDDEN -> PCA_DIM. Input is |psi|^2 (32 real,
    non-negative measurement probabilities, sums to 1), NOT the raw
    complex state -- decoders in this project are classical, real-valued.
    Reconstructs the (frozen) PCA output, same convention as
    Group6/Group14.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, DEC_HIDDEN), nn.ReLU(),
            nn.Linear(DEC_HIDDEN, PCA_DIM),
        )

    def forward(self, probs):
        return self.net(probs)


class VQCEncoder(nn.Module):
    """
    Full model: frozen PCA -> VQC (200 trainable params) -> full 32-dim
    complex state. Decoder reconstructs the PCA output from |psi|^2.
    """
    def __init__(self, input_dim, seed=42):
        super().__init__()
        set_seed(seed)
        self.input_dim = input_dim
        self.pca_comp  = PCACompressor(input_dim)
        self.decoder   = Decoder()
        self.q_weights = nn.Parameter(torch.randn(N_LAYERS, N_QUBITS, 2) * 0.01)
        self.circuit   = make_state_circuit(N_QUBITS, N_LAYERS)

    def compress(self, x):
        """Raw x -> frozen PCA -> L2-normalized 32-dim (NOT trainable)."""
        return self.pca_comp.compress_tensor(x)

    def get_states(self, z_norm_batch):
        """z_norm_batch: [B, 32] -> [B, 32] complex state vectors."""
        out = []
        for z in z_norm_batch:
            psi = self.circuit(z, self.q_weights)
            out.append(psi)
        return torch.stack(out)

    def forward(self, x):
        """Returns (states, x_hat, z_pca)."""
        z_pca  = self.compress(x)
        states = self.get_states(z_pca)
        probs  = states.abs() ** 2   # [B, 32] real, non-negative, sums to 1
        x_hat  = self.decoder(probs)
        return states, x_hat, z_pca

    def get_states_numpy(self, X, batch_size=64):
        self.eval()
        Xt  = torch.FloatTensor(X.astype(np.float32))
        out = []
        with torch.no_grad():
            for s in range(0, len(Xt), batch_size):
                z_pca = self.compress(Xt[s:s + batch_size])
                psi   = self.get_states(z_pca)
                out.append(psi.detach().cpu().numpy())
        return np.vstack(out)


# ======================================================================
# LOSS FUNCTION: fixed-target fidelity
# ======================================================================

def fixed_target_fidelity_loss(states, labels):
    """
    L = mean( 1 - |<target_y_i | psi_i>|^2 )

    target_c is the computational basis state |c> (c = 0..N_CLASSES-1),
    fixed, NEVER updated by training. Since target_c is one-hot,
    <target_c|psi> = psi[c] directly -- fidelity to a fixed basis-state
    target is just |psi[c]|^2 (verified numerically: matches the full
    inner-product formula to floating-point precision).

    Unlike CADE's pairwise contrastive loss, no explicit "push apart"
    term is needed: the targets are pairwise ORTHOGONAL, so routing a
    sample toward fidelity~1 with its own target automatically forces
    near-zero fidelity with every other target.

    No left/right batch pairing needed either -- ordinary shuffled
    minibatches, one term per sample.
    """
    idx = labels.long()
    B   = states.shape[0]
    own_amplitude = states[torch.arange(B), idx]   # complex, [B]
    fid = (own_amplitude.abs()) ** 2                 # real, [B]
    return (1.0 - fid).mean()


def reconstruction_loss(x_hat, x):
    return nn.functional.mse_loss(x_hat, x)


def epoch_batches_simple(X_train, y_train, batch_size):
    """
    Ordinary full-epoch sweep: shuffle the training set once, split into
    batch_count non-overlapping chunks of batch_size (dropping any
    remainder), covering (close to) the full training set in one epoch.
    No CADE-style same/different-class pairing needed here -- this loss
    scores each sample against its own FIXED target directly, not
    against other samples, so there is no pairing structure to build.

    Fixes the same underlying bug found in Group14/Group15/M0a: the
    previous make_simple_batch drew exactly ONE random batch per
    "epoch" iteration, so CAE_EPOCHS*CAE_BATCH_SIZE=12,800 total
    training instances were seen across the ENTIRE run, regardless of
    actual dataset size -- not a real epoch sweep at all.

    Returns (batch_count, X_batches, y_batches), each of shape
    [batch_count, batch_size, ...].
    """
    n_train = X_train.shape[0]
    batch_count = n_train // batch_size
    if batch_count == 0:
        raise ValueError(f'n_train={n_train} smaller than batch_size={batch_size}')

    random_idx = np.random.permutation(n_train)
    used_idx = random_idx[: batch_count * batch_size]

    X_batches = np.split(X_train[used_idx], batch_count)
    y_batches = np.split(y_train[used_idx], batch_count)

    return batch_count, np.array(X_batches), np.array(y_batches)


def train_model(model, X_train, y_train, seed, device='cpu'):
    model.to(device)

    print(f'\n  [STAGE 1a] Fitting PCA: F={X_train.shape[1]}->{PCA_DIM}', flush=True)
    model.pca_comp.fit(X_train)

    trainable_params = [model.q_weights] + list(model.decoder.parameters())
    optimizer = optim.Adam(trainable_params, lr=CAE_LR)

    n_params = sum(p.numel() for p in trainable_params if p.requires_grad)
    print(f'\n  [STAGE 1b] Training VQC (fixed-target fidelity)', flush=True)
    print(f'  Quantum params: {model.q_weights.numel()}  '
          f'Decoder params: {sum(p.numel() for p in model.decoder.parameters())}  '
          f'Total trainable (quantum only, for capacity comparison): {model.q_weights.numel()}', flush=True)
    print(f'  Circuit: {N_QUBITS} qubits x {N_LAYERS} layers, '
          f'readout=full state ({STATE_DIM}-dim complex)', flush=True)
    print(f'  Targets: computational basis states |0>..|{N_CLASSES-1}> (fixed, '
          f'pairwise orthogonal, never trained)', flush=True)
    print(f'  Loss: mean(1 - fidelity(psi, target_y)) + {RECON_WEIGHT}*reconstruction', flush=True)
    print(f'  LR: {CAE_LR}  Max epochs: {CAE_EPOCHS}  Patience: {PATIENCE}  RelMinDelta: {REL_MIN_DELTA}  '
          f'(full-epoch sweep, no single-batch-per-epoch bug)', flush=True)

    model.train()
    set_seed(seed)

    best_loss = float('inf')
    best_epoch = 0
    best_state = None             # weights at the best-loss epoch (restored at the end)
    epochs_since_improve = 0
    stopped_early = False
    actual_epochs = CAE_EPOCHS

    for epoch in range(1, CAE_EPOCHS + 1):
        batch_count, X_batches, y_batches = epoch_batches_simple(
            X_train, y_train, CAE_BATCH_SIZE)

        loss_list, fid_list, recon_list = [], [], []
        for b in range(batch_count):
            Xb = torch.FloatTensor(X_batches[b]).to(device)
            yb = torch.LongTensor(y_batches[b]).to(device)

            optimizer.zero_grad()
            z_pca_b = model.compress(Xb)
            states  = model.get_states(z_pca_b)
            probs   = states.abs() ** 2
            x_hat   = model.decoder(probs)

            l_fid   = fixed_target_fidelity_loss(states, yb)
            l_recon = reconstruction_loss(x_hat, z_pca_b)
            loss    = l_fid + RECON_WEIGHT * l_recon

            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            fid_list.append(l_fid.item())
            recon_list.append(l_recon.item())

        epoch_loss = float(np.mean(loss_list))

        # Relative min_delta: an epoch counts as an improvement only if it beats
        # best_loss by more than REL_MIN_DELTA * best_loss. Scales with each
        # model's own loss magnitude. best_loss=inf on epoch 1 -> always improves.
        if epoch_loss < best_loss * (1.0 - REL_MIN_DELTA):
            best_loss = epoch_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        # Logged EVERY epoch: PATIENCE=5 with no min-epoch floor can stop as early
        # as epoch 6, and the per-epoch loss curve is needed for the LR diagnosis.
        print(f'  Epoch {epoch:3d}  batches={batch_count}  '
              f'total={epoch_loss:.5f}  '
              f'fid_loss={np.mean(fid_list):.5f}  '
              f'recon={np.mean(recon_list):.5f}  '
              f'best={best_loss:.5f}  no_improve={epochs_since_improve}', flush=True)

        if epochs_since_improve >= PATIENCE:
            print(f'\n  [EARLY STOP] No improvement for {PATIENCE} epochs '
                  f'(best_loss={best_loss:.5f} at epoch {epoch - epochs_since_improve}). '
                  f'Stopping at epoch {epoch}.', flush=True)
            stopped_early = True
            actual_epochs = epoch
            break

    if not stopped_early:
        print(f'\n  [MAX EPOCHS] Reached CAE_EPOCHS={CAE_EPOCHS} without triggering '
              f'early stop (best_loss={best_loss:.5f}).', flush=True)

    # Restore the BEST-loss weights (not the last epoch's) -- this is the
    # model that gets saved and scored.
    model.load_state_dict(best_state)
    print(f'  [RESTORE] Using weights from best epoch {best_epoch} '
          f'(best_loss={best_loss:.5f}), not final epoch {actual_epochs}.', flush=True)

    return model, actual_epochs, stopped_early, best_epoch, best_loss


# ======================================================================
# EVALUATION -- fixed-target fidelity scoring (targets are FIXED
# computational basis states, never computed from training data, unlike
# every other script's data-derived centroids)
# ======================================================================

def compute_fidelity_target_scores(states_train, y_train, states_test, n_classes):
    """
    MAD-based OOD scoring using fixed-target fidelity as 'distance'.

    distance_to_class_c = 1 - fidelity_to_target_c = 1 - |psi[c]|^2

    Mirrors compute_mad_scores' exact structure/semantics (min distance
    = best-matching-class distance for threshold-free scoring; MAD-
    normalized version for M0a-style threshold scoring) -- the only
    difference is 'distance to a FIXED target' instead of 'Euclidean
    distance to a data-derived centroid'.
    """
    probs_train = np.abs(states_train) ** 2   # [N_train, STATE_DIM]
    probs_test  = np.abs(states_test)  ** 2   # [N_test,  STATE_DIM]

    # Score ONLY against the KNOWN classes' targets (labels present in
    # y_train). The held-out OOD family's label index must never be used:
    # iterating range(n_classes) would include the OOD class's basis state
    # (label leakage) and an empty-class median (NaN -> every MAD score NaN
    # -> MAD F1 = 0). Same convention as Group14/Group15, whose centroids
    # are built only from classes present in y_train.
    classes = sorted(int(c) for c in np.unique(y_train))
    class_stats = {}
    for c in classes:
        idx_c  = (y_train == c)
        dist_c = 1.0 - probs_train[idx_c, c]   # class-c samples' distance to THEIR OWN target
        med    = np.median(dist_c)
        mad    = 1.4826 * np.median(np.abs(dist_c - med))
        mad    = max(mad, 1e-6)
        class_stats[c] = {'median': med, 'mad': mad}

    raw_scores, mad_scores = [], []
    for i in range(probs_test.shape[0]):
        dist_k = [1.0 - probs_test[i, c] for c in classes]
        anom_k = [abs(dist_k[j] - class_stats[c]['median']) / class_stats[c]['mad']
                  for j, c in enumerate(classes)]
        raw_scores.append(float(np.min(dist_k)))
        mad_scores.append(float(np.min(anom_k)))

    return np.array(raw_scores), np.array(mad_scores)


def evaluate_ood(scores, y_test, ood_label):
    n_ood   = int(np.sum(y_test == ood_label))
    n_known = len(y_test) - n_ood
    n_total = len(y_test)
    order   = np.argsort(scores)[::-1]
    y_s     = y_test[order]

    pc, rc, f1c = [], [], []
    tp = 0
    for k in range(1, n_total + 1):
        if y_s[k - 1] == ood_label:
            tp += 1
        p = tp / k
        r = tp / n_ood if n_ood > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        pc.append(p); rc.append(r); f1c.append(f)

    f1_arr   = np.asarray(f1c)
    best_idx = int(np.argmax(f1_arr))
    effort   = (best_idx + 1) / n_ood if n_ood > 0 else 0.0
    top_k    = order[:best_idx + 1]
    tp_v     = int(np.sum(y_test[top_k] == ood_label))
    fp_v     = (best_idx + 1) - tp_v
    fn_v     = n_ood - tp_v
    fpr      = fp_v / n_known if n_known > 0 else 0.0
    fnr      = fn_v / n_ood   if n_ood   > 0 else 0.0

    tpr_r, fpr_r = [0.0], [0.0]
    tp_r, fp_r   = 0, 0
    for i in range(n_total):
        if y_test[order[i]] == ood_label:
            tp_r += 1
        else:
            fp_r += 1
        tpr_r.append(tp_r / n_ood   if n_ood   > 0 else 0.0)
        fpr_r.append(fp_r / n_known if n_known > 0 else 0.0)

    tpr_roc = np.asarray(tpr_r)
    fpr_roc = np.asarray(fpr_r)
    roc_auc = float(np.clip(np.trapezoid(tpr_roc, fpr_roc), 0.0, 1.0))
    rec_a   = np.concatenate([[0.0], np.asarray(rc)])
    prec_a  = np.concatenate([[1.0], np.asarray(pc)])
    pr_auc  = float(np.clip(np.trapezoid(prec_a, rec_a), 0.0, 1.0))

    return {
        "best_f1":        float(f1_arr[best_idx]),
        "best_precision": float(pc[best_idx]),
        "best_recall":    float(rc[best_idx]),
        "norm_effort":    float(effort),
        "fpr":            float(fpr),
        "fnr":            float(fnr),
        "roc_auc":        roc_auc,
        "pr_auc":         pr_auc,
        "n_ood":          int(n_ood),
        "n_known":        int(n_known),
        "n_total":        int(n_total),
    }


def evaluate_mad_threshold(mad_scores, y_test, ood_label, threshold=MAD_THRESHOLD):
    predicted_ood = mad_scores > threshold
    actual_ood    = y_test == ood_label
    n_ood         = int(actual_ood.sum())
    n_known       = len(y_test) - n_ood

    tp = int((predicted_ood & actual_ood).sum())
    fp = int((predicted_ood & ~actual_ood).sum())
    fn = int((~predicted_ood & actual_ood).sum())
    tn = int((~predicted_ood & ~actual_ood).sum())

    prec   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1     = 2*prec*recall/(prec+recall) if (prec+recall) > 0 else 0.0
    fpr    = fp / n_known if n_known > 0 else 0.0
    fnr    = fn / n_ood   if n_ood   > 0 else 0.0
    effort = (tp + fp) / n_ood if n_ood > 0 else 0.0

    return {
        'threshold':      float(threshold),
        'best_f1':        round(f1,     4),
        'best_precision': round(prec,   4),
        'best_recall':    round(recall, 4),
        'fpr':            round(fpr,    4),
        'fnr':            round(fnr,    4),
        'norm_effort':    round(effort, 4),
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
    }


# ======================================================================
# SAVE RESULTS
# ======================================================================

def save_results(save_dir, run_id, split_id, seed, n_train, n_test,
                 n_features, train_time, actual_epochs, stopped_early,
                 best_epoch, best_loss,
                 mad_res, tf_res, raw_scores, mad_scores, ood_family_name="unknown"):

    summary = {
        'model':          'Group17_VQC_FixedTargetFidelity_4class',
        'reference':      'Fixed-target fidelity: VQC (5 qubits, 20 layers, 200 params) '
                           'trained via L=1-fidelity(psi,target_y), targets = fixed, '
                           'pairwise-orthogonal computational basis states, never trained',
        'architecture':   f'{n_features}->PCA({PCA_DIM})->VQC({N_QUBITS}q,{N_LAYERS}L)->'
                           f'full_state({STATE_DIM})  ({N_QUANTUM_PARAMS} quantum trainable params)',
        'loss':           f'mean(1-fidelity(psi,target_y)) + {RECON_WEIGHT}*reconstruction',
        'run_id':         int(run_id),
        'split_id':       int(split_id),
        'ood_family':     ood_family_name,
        'seed':           int(seed),
        'n_train':        int(n_train),
        'n_test':         int(n_test),
        'n_features':     int(n_features),
        'n_classes':      N_CLASSES,
        'state_dim':      STATE_DIM,
        'n_qubits':       N_QUBITS,
        'n_layers':       N_LAYERS,
        'n_quantum_params': N_QUANTUM_PARAMS,
        'recon_weight':   RECON_WEIGHT,
        'train_time_sec': round(train_time, 2),
        'max_epochs':     CAE_EPOCHS,
        'actual_epochs':  int(actual_epochs),
        'stopped_early':  bool(stopped_early),
        'best_epoch':     int(best_epoch),       # weights saved/scored come from this epoch
        'best_loss':      float(best_loss),
        'lr':             CAE_LR,
        'patience':       PATIENCE,
        'rel_min_delta':  REL_MIN_DELTA,
        'MAD_Threshold':  mad_res,   # MAD-based thresholding, CADE's SCORING
                                     # PROTOCOL reused -- this is not CADE itself
        'Threshold_Free': tf_res,   # same protocol note applies
    }

    with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    import csv
    with open(os.path.join(save_dir, 'ood_scores.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['raw_score', 'mad_score'])
        for r, m in zip(raw_scores, mad_scores):
            w.writerow([r, m])

    print(f'  Saved: {save_dir}', flush=True)


# ======================================================================
# RUN TASK
# ======================================================================

def run_task(task_id, data_dir, prefix, result_dir):
    run_id   = task_id // N_SPLITS
    split_id = task_id %  N_SPLITS
    seed     = SEEDS[run_id]

    _npz_check = os.path.join(os.path.expanduser(data_dir), f'{prefix}_{split_id}.npz')
    _d_check   = np.load(_npz_check, allow_pickle=True)
    ood_family_name = str(_d_check['heldout_family'].item()) \
                      if _d_check['heldout_family'].shape == () \
                      else str(_d_check['heldout_family'])

    print('=' * 65, flush=True)
    print(f'Group17 VQC Fixed-Target Fidelity (4-class) | Task={task_id} | Run={run_id} | '
          f'Split={split_id} | Seed={seed} | OOD={ood_family_name}', flush=True)
    print('=' * 65, flush=True)

    save_dir = os.path.join(result_dir, f'run{run_id}', f'split{split_id}')
    os.makedirs(save_dir, exist_ok=True)

    npz_path = os.path.join(data_dir, f'{prefix}_{split_id}.npz')
    print(f'  Loading: {npz_path}', flush=True)
    d = np.load(npz_path, allow_pickle=True)
    _xt = d['X_train']
    X_train = _xt.item().toarray().astype(np.float32) if _xt.shape == () else _xt.astype(np.float32)
    y_train = d['y_train'].astype(np.int64)
    _xte = d['X_test']
    X_test  = _xte.item().toarray().astype(np.float32) if _xte.shape == () else _xte.astype(np.float32)
    y_test  = d['y_test'].astype(np.int64)

    print(f'  X_train={X_train.shape}  X_test={X_test.shape}', flush=True)
    print(f'  Classes in y_train={len(np.unique(y_train))} '
          f'(OOD family {ood_family_name} absent — expected; {N_CLASSES}-class subset)', flush=True)

    set_seed(seed)

    print(f'\n[STAGE 1] Training VQC (fixed-target fidelity)', flush=True)
    model = VQCEncoder(X_train.shape[1], seed=seed)
    t0 = time.time()
    model, actual_epochs, stopped_early, best_epoch, best_loss = train_model(
        model, X_train, y_train, seed)
    train_time = time.time() - t0
    print(f'  Training time: {train_time:.1f}s  '
          f'(ran {actual_epochs}/{CAE_EPOCHS} epochs, '
          f'early_stopped={stopped_early}, best_epoch={best_epoch})', flush=True)

    weights_path = os.path.join(save_dir, 'model_weights.pt')
    torch.save(model.state_dict(), weights_path)
    import pickle
    with open(os.path.join(save_dir, 'pca_components.pkl'), 'wb') as f:
        pickle.dump(model.pca_comp.pca, f)

    print(f'\n[STAGE 2] Extracting full quantum states', flush=True)
    states_train = model.get_states_numpy(X_train)
    states_test  = model.get_states_numpy(X_test)
    print(f'  states_train={states_train.shape}  states_test={states_test.shape}', flush=True)

    print(f'\n[STAGE 3] OOD scoring (CADE MAD + threshold-free, fixed-target fidelity)', flush=True)
    raw_scores, mad_scores = compute_fidelity_target_scores(
        states_train, y_train, states_test, N_CLASSES)

    known = set(np.unique(y_train).tolist())
    candidates = set(np.unique(y_test).tolist()) - known
    ood_label = candidates.pop() if len(candidates) == 1 else None
    if ood_label is None:
        raise RuntimeError(f"Expected exactly one OOD label, found: {candidates}")

    mad_res = evaluate_mad_threshold(mad_scores, y_test, ood_label)
    tf_res  = evaluate_ood(raw_scores, y_test, ood_label)

    print(f'  [MAD]  F1={mad_res["best_f1"]:.4f}  FPR={mad_res["fpr"]:.4f}', flush=True)
    print(f'  [ThreshFree] F1={tf_res["best_f1"]:.4f}  ROC_AUC={tf_res["roc_auc"]:.4f}', flush=True)

    save_results(save_dir, run_id, split_id, seed,
                 X_train.shape[0], X_test.shape[0], X_train.shape[1],
                 train_time, actual_epochs, stopped_early,
                 best_epoch, best_loss,
                 mad_res, tf_res, raw_scores, mad_scores,
                 ood_family_name=ood_family_name)

    print(f'\n[DONE] OOD={ood_family_name}  MAD F1={mad_res["best_f1"]:.4f}  '
          f'ThreshFree F1={tf_res["best_f1"]:.4f}', flush=True)


# ======================================================================
# AGGREGATE
# ======================================================================

@_tee_to_file('aggregate_summary.txt')
def aggregate(result_dir):
    rows, missing = [], []

    n_splits = 0
    for _s in range(20):
        if os.path.exists(os.path.join(result_dir, 'run0', f'split{_s}', 'summary.json')):
            n_splits = _s + 1
        else:
            break
    if n_splits == 0:
        print('No results found.')
        return

    for run_id in range(3):
        for split_id in range(n_splits):
            path = os.path.join(result_dir, f'run{run_id}', f'split{split_id}', 'summary.json')
            if not os.path.exists(path):
                missing.append((run_id, split_id))
                continue
            with open(path) as f:
                d = json.load(f)
            rows.append({
                'run_id':     d.get('run_id', run_id),
                'split_id':   d.get('split_id', split_id),
                'ood_family': d.get('ood_family', f'split{split_id}'),
                'seed':       d.get('seed', SEEDS[run_id]),
                'mad__F1':    d.get('MAD_Threshold', {}).get('best_f1', float('nan')),
                'tf__F1':     d.get('Threshold_Free', {}).get('best_f1', float('nan')),
                'tf__ROC_AUC':d.get('Threshold_Free', {}).get('roc_auc', float('nan')),
                'tf__FPR':    d.get('Threshold_Free', {}).get('fpr', float('nan')),
            })

    if missing:
        print(f'Missing {len(missing)}/{3 * n_splits}: {missing}')
    if not rows:
        print('No results found.')
        return

    def mat(col):
        out = np.full((3, n_splits), np.nan)
        for r in rows:
            out[int(r['run_id']), int(r['split_id'])] = r.get(col, np.nan)
        return out

    split_family = {}
    for r in rows:
        split_family.setdefault(int(r['split_id']), r['ood_family'])

    mad_f1 = mat('mad__F1')
    tf_f1  = mat('tf__F1')
    tf_roc = mat('tf__ROC_AUC')

    print('=' * 72)
    print('Group17 VQC FIXED-TARGET FIDELITY (4-class) — AGGREGATED RESULTS')
    print('=' * 72)
    print(f'Completed: {len(rows)}/{3 * n_splits}')
    print()
    print(f'  {"Family":<16} {"MAD_F1":>8} {"TF_F1":>8} {"TF_ROC_AUC":>11}')
    print(f'  {"-"*47}')
    for s in range(n_splits):
        print(f'  {split_family.get(s, f"split{s}"):<16}'
              f'{np.nanmean(mad_f1[:, s]):>8.4f}'
              f'{np.nanmean(tf_f1[:, s]):>8.4f}'
              f'{np.nanmean(tf_roc[:, s]):>11.4f}')
    print(f'  {"-"*47}')
    print(f'  {"MEAN":<16}{np.nanmean(mad_f1):>8.4f}{np.nanmean(tf_f1):>8.4f}{np.nanmean(tf_roc):>11.4f}')
    print(f'  {"STD":<16}{np.nanstd(mad_f1):>8.4f}{np.nanstd(tf_f1):>8.4f}{np.nanstd(tf_roc):>11.4f}')

    final = {
        'model': 'Group17_VQC_FixedTargetFidelity_4class',
        'n_results': len(rows),
        'expected': 3 * n_splits,
        'MAD_Threshold':  {'F1_mean': round(float(np.nanmean(mad_f1)), 4),
                                 'F1_std':  round(float(np.nanstd(mad_f1)), 4)},
        'Threshold_Free': {'F1_mean': round(float(np.nanmean(tf_f1)), 4),
                                 'F1_std':  round(float(np.nanstd(tf_f1)), 4),
                                 'ROC_AUC_mean': round(float(np.nanmean(tf_roc)), 4)},
    }
    with open(os.path.join(result_dir, 'final_summary.json'), 'w') as f:
        json.dump(final, f, indent=2)
    print(f'\nSaved: {os.path.join(result_dir, "final_summary.json")}')


# ======================================================================
# MAIN
# ======================================================================

def main():
    p = argparse.ArgumentParser(
        description='Group17: VQC fixed-target fidelity (5q, 20L, full-state) — 4-class LAMDA')
    p.add_argument('task_id', help="Integer 0-11 for a specific split, or 'aggregate'")
    p.add_argument('--data-dir', default=DEFAULT_DATA_DIR)
    p.add_argument('--prefix', default='drebin_new')
    p.add_argument('--results-dir', default=DEFAULT_RESULTS_DIR)
    args = p.parse_args()

    if args.task_id == 'aggregate':
        aggregate(os.path.expanduser(args.results_dir))
    else:
        task_id = int(args.task_id)
        if not (0 <= task_id <= 3 * N_SPLITS - 1):
            raise ValueError(f'task_id must be 0-{3*N_SPLITS-1} for {N_SPLITS} splits x 3 seeds')
        run_task(task_id, os.path.expanduser(args.data_dir), args.prefix,
                  os.path.expanduser(args.results_dir))


if __name__ == '__main__':
    main()
