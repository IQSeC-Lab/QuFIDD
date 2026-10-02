#!/usr/bin/env python3
"""
Group14_vqc_m1a_fix_4class.py
================================
"M1a-fix": a VQC variant designed to escape both structural traps found
in the G1/G6 fidelity-based scoring (M1b/M1c), which we proved are
mathematically invariant to the VQC's parameters (unitary-invariance of
full-state inner products), and the reduced-density-matrix alternative
(M1c'), which we showed is biased in the wrong direction by fidelity's
monotonicity under partial trace.

Design:
  - Readout is the FULL local Bloch vector per qubit: <X>,<Y>,<Z> for
    each of 5 qubits = 15-dim real vector. Each <P_i> = <psi|P_i|psi>
    is a genuinely theta-dependent quadratic functional (NOT an inner
    product between two theta-evolved states), so it is NOT covered by
    the unitary-invariance proof, and it is not a fidelity/overlap
    quantity, so the monotonicity bound doesn't apply either.
  - Loss and score are the IDENTICAL 15-dim quantity, using CADE's own
    exact Euclidean contrastive loss formula (unsquared distance,
    same-class pull, different-class relu(margin-dist) push,
    margin=10.0, lambda=0.1) -- reused verbatim from M0a_CADE_original.py.
    This fixes the loss/score mismatch that made the ORIGINAL M1a
    (Pauli-Z-only, fidelity-trained) weak.
  - 5 qubits, 20 layers (not 30) => 5*20*2 = 200 trainable quantum params.
  - Frozen PCA(F->32) front-end, IDENTICAL PCACompressor class to
    Group6/Group9/Group13 -- same role, same L2-normalized output.
  - Decoder reconstructs the PCA output (not raw x), same convention as
    Group6/Group13, since PCA is frozen and provides no gradient path.

Capacity-matched counterpart: Group15_cade_mini_4class.py (~215 params,
same PCA front-end, CADE's own architecture shrunk to n_classes-1=3).

Data: 4-class LAMDA subset (droidkungfu, plankton, smsreg, airpush --
the four families that hit the full 1000-sample cap), built by
filter_lamda_to_4class.py + build_cade_splits_monthwise.py +
apply_cade_variance_threshold.py (all unchanged from the 8-class pipeline).

Usage:
    python3 Group14_vqc_m1a_fix_4class.py <task_id> \\
        --data-dir <path to 4-class cade_lamda_vt> \\
        --prefix drebin_new \\
        --results-dir results

    python3 Group14_vqc_m1a_fix_4class.py aggregate --results-dir results
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

PCA_DIM         = 32                # frozen PCA output dim -- same as Group6/Group9/Group13
N_QUBITS        = 5
N_LAYERS        = 20                # NOT 30 -- reduced per this experiment's spec
N_QUANTUM_PARAMS = N_QUBITS * N_LAYERS * 2   # = 200
READOUT_DIM     = N_QUBITS * 3      # <X>,<Y>,<Z> per qubit = 15

DEC_HIDDEN      = 24                # decoder hidden width (uncapped -- decoder params
                                     # are not part of the 200-param quantum budget,
                                     # same convention as Group6/Group13's decoders)

# CADE's own exact hyperparameters, except CAE_LR (see below). CAE_EPOCHS
# and SIMILAR_RATIO corrected below to match the confirmed real CADE
# defaults (were 200 and 0.5).
CAE_LR          = 0.001              # raised from CADE's 0.0001: job 6839482 at 0.0001
                                     # barely moved from its near-identity init (con loss
                                     # 4.93->4.84, ROC-AUC 0.49); same value as Group15
CAE_BATCH_SIZE  = 64
CAE_EPOCHS      = 250                # matches real CADE's confirmed default (was 200) --
                                     # now a CEILING, not a fixed count; early stopping below
PATIENCE        = 5                  # epochs with no improvement before stopping
REL_MIN_DELTA   = 1e-4               # relative improvement threshold: must beat
                                     # best_loss by > REL_MIN_DELTA * best_loss
                                     # (same rule as Group15/Group17)
SIMILAR_RATIO   = 0.25               # matches real CADE's confirmed default (was 0.5) --
                                     # combined with epoch_batches' num_sim=batch_size*ratio
                                     # formula, gives the correct 50/50 same/diff split
LAMBDA_1        = 0.1
MARGIN          = 10.0
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
# VQC: 5 qubits, 20 layers, full XYZ readout per qubit
# ======================================================================

def make_xyz_circuit(n_qubits, n_layers):
    """
    Amplitude-embed a normalized PCA_DIM-vector into n_qubits, apply the
    same hardware-efficient ansatz used throughout this project (RY+RZ
    per qubit per layer, ring CNOT + alternating long-range CNOT), then
    read out <X>,<Y>,<Z> for EVERY qubit (not just <Z>) -- 3*n_qubits
    real values total.
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
        obs = ([qml.expval(qml.PauliX(i)) for i in range(n_qubits)] +
               [qml.expval(qml.PauliY(i)) for i in range(n_qubits)] +
               [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)])
        return obs

    return circuit


class Decoder(nn.Module):
    """READOUT_DIM -> DEC_HIDDEN -> PCA_DIM. Reconstructs the (frozen)
    PCA output, not raw x -- same convention as Group6/Group13."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(READOUT_DIM, DEC_HIDDEN), nn.ReLU(),
            nn.Linear(DEC_HIDDEN, PCA_DIM),
        )

    def forward(self, q):
        return self.net(q)


class VQCEncoder(nn.Module):
    """
    Full model: frozen PCA -> VQC (200 trainable params) -> full XYZ
    readout (15-dim). Decoder reconstructs the PCA output.
    """
    def __init__(self, input_dim, seed=42):
        super().__init__()
        set_seed(seed)
        self.input_dim = input_dim
        self.pca_comp  = PCACompressor(input_dim)
        self.decoder   = Decoder()
        self.q_weights = nn.Parameter(torch.randn(N_LAYERS, N_QUBITS, 2) * 0.01)
        self.circuit   = make_xyz_circuit(N_QUBITS, N_LAYERS)

    def compress(self, x):
        """Raw x -> frozen PCA -> L2-normalized 32-dim (NOT trainable)."""
        return self.pca_comp.compress_tensor(x)

    def readout(self, z_norm_batch):
        """z_norm_batch: [B, 32] -> [B, 15] full XYZ expectation values."""
        out = []
        for z in z_norm_batch:
            vals = self.circuit(z, self.q_weights)
            out.append(torch.stack(vals))
        return torch.stack(out).float()

    def forward(self, x):
        """Returns (q_xyz, x_hat, z_pca)."""
        z_pca = self.compress(x)
        q_xyz = self.readout(z_pca)
        x_hat = self.decoder(q_xyz)
        return q_xyz, x_hat, z_pca

    def get_readout_numpy(self, X, batch_size=64):
        self.eval()
        Xt  = torch.FloatTensor(X.astype(np.float32))
        out = []
        with torch.no_grad():
            for s in range(0, len(Xt), batch_size):
                z_pca = self.compress(Xt[s:s + batch_size])
                q     = self.readout(z_pca)
                out.append(q.detach().cpu().numpy())
        return np.vstack(out)


# ======================================================================
# LOSS FUNCTIONS (verbatim from M0a_CADE_original.py -- CADE's exact
# formula, unmodified, applied to the 15-dim readout instead of CADE's
# own 7-dim bottleneck)
# ======================================================================

def contrastive_loss(z, labels):
    """
    EXACT CADE contrastive loss from autoencoder.py.
    Batch layout: first half = left, second half = right.
    is_same = (labels[left] == labels[right])
    dist = ||z_left - z_right||_2
    loss = is_same*dist + (1-is_same)*relu(margin - dist)
    NOT squared distances. relu not squared. margin=10.0 unchanged --
    CADE's own real 7-dim bottleneck is ALSO Tanh-bounded with this same
    "unreachable" margin (max achievable distance ~5.29 < 10), and that
    is CADE's real, published, working design -- not a bug to fix here.
    """
    half    = z.shape[0] // 2
    z_left  = z[:half]
    z_right = z[half:]
    y_left  = labels[:half]
    y_right = labels[half:]
    diff    = z_left - z_right
    dist    = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-10)
    is_same = (y_left == y_right).float()
    loss    = is_same * dist + (1.0 - is_same) * torch.relu(MARGIN - dist)
    return loss.mean()


def reconstruction_loss(x_hat, x):
    return nn.functional.mse_loss(x_hat, x)


def epoch_batches(X_train, y_train, batch_size, similar_ratio):
    """
    Faithful port of the REAL CADE epoch_batches() from cade/data.py
    (confirmed via direct source inspection, github.com/S-R-Nowmi/CADE).

    Sweeps the FULL training set once per epoch: a single random
    permutation of X_train is split into batch_count non-overlapping
    chunks of half_size, so every training sample is used exactly once
    per epoch as a pairing anchor (the left half of some batch). The
    right half of each batch is filled per-anchor with a same-class
    partner (num_sim of them) or different-class partner (the rest) --
    num_sim = int(batch_size * similar_ratio), matching the real
    formula exactly (NOT half_size * similar_ratio, which the earlier
    make_cade_batch incorrectly used).

    Returns (batch_count, b_out_x, b_out_y), shape
    [batch_count, batch_size, ...] -- iterate b in range(batch_count)
    for one epoch's worth of batches.
    """
    if batch_size % 4 != 0:
        raise ValueError('batch_size should be a multiple of 4.')
    half_size = batch_size // 2

    n_train = X_train.shape[0]
    batch_count = int(n_train / half_size)
    if batch_count == 0:
        raise ValueError(f'n_train={n_train} too small for half_size={half_size}')

    num_sim = int(batch_size * similar_ratio)

    b_out_x = np.zeros([batch_count, batch_size, X_train.shape[1]], dtype=np.float32)
    b_out_y = np.zeros([batch_count, batch_size], dtype=np.int64)

    random_idx = np.random.permutation(n_train)
    b_out_x[:, :half_size] = np.split(X_train[random_idx[: batch_count * half_size]], batch_count)
    b_out_y[:, :half_size] = np.split(y_train[random_idx[: batch_count * half_size]], batch_count)

    classes = np.unique(y_train)
    max_label = int(classes.max())
    index_cls    = [None] * (max_label + 1)
    index_no_cls = [None] * (max_label + 1)
    for label in classes:
        label = int(label)
        index_cls[label]    = np.where(y_train == label)[0]
        index_no_cls[label] = np.where(y_train != label)[0]

    for b in range(batch_count):
        for m in range(0, num_sim):
            lbl = int(b_out_y[b, m])
            pair = np.random.choice(index_cls[lbl], 1)
            b_out_x[b, m + half_size] = X_train[pair[0]]
            b_out_y[b, m + half_size] = y_train[pair[0]]
        for m in range(num_sim, half_size):
            lbl = int(b_out_y[b, m])
            pair = np.random.choice(index_no_cls[lbl], 1)
            b_out_x[b, m + half_size] = X_train[pair[0]]
            b_out_y[b, m + half_size] = y_train[pair[0]]

    return batch_count, b_out_x, b_out_y


def train_model(model, X_train, y_train, seed, device='cpu'):
    model.to(device)

    print(f'\n  [STAGE 1a] Fitting PCA: F={X_train.shape[1]}->{PCA_DIM}', flush=True)
    model.pca_comp.fit(X_train)

    trainable_params = [model.q_weights] + list(model.decoder.parameters())
    optimizer = optim.Adam(trainable_params, lr=CAE_LR)

    n_params = sum(p.numel() for p in trainable_params if p.requires_grad)
    print(f'\n  [STAGE 1b] Training VQC (M1a-fix, XYZ readout)', flush=True)
    print(f'  Quantum params: {model.q_weights.numel()}  '
          f'Decoder params: {sum(p.numel() for p in model.decoder.parameters())}  '
          f'Total trainable (quantum only, for capacity comparison): {model.q_weights.numel()}', flush=True)
    print(f'  Circuit: {N_QUBITS} qubits x {N_LAYERS} layers, '
          f'readout=<X,Y,Z> per qubit ({READOUT_DIM}-dim)', flush=True)
    print(f'  Loss: {LAMBDA_1}*contrastive(Euclidean,margin={MARGIN}) + reconstruction '
          f'(CADE\'s exact formula, on the {READOUT_DIM}-dim readout)', flush=True)
    print(f'  LR: {CAE_LR}  Max epochs: {CAE_EPOCHS}  Patience: {PATIENCE}  RelMinDelta: {REL_MIN_DELTA}  '
          f'(full-epoch sweep, faithful to real CADE\'s epoch_batches)', flush=True)

    model.train()
    set_seed(seed)

    best_loss = float('inf')
    best_epoch = 0
    best_state = None             # weights at the best-loss epoch (restored at the end)
    epochs_since_improve = 0
    stopped_early = False
    actual_epochs = CAE_EPOCHS

    for epoch in range(1, CAE_EPOCHS + 1):
        batch_count, b_out_x, b_out_y = epoch_batches(
            X_train, y_train, CAE_BATCH_SIZE, SIMILAR_RATIO)

        loss_list, con_list, recon_list = [], [], []
        for b in range(batch_count):
            Xb = torch.FloatTensor(b_out_x[b]).to(device)
            yb = torch.LongTensor(b_out_y[b]).to(device)

            optimizer.zero_grad()
            z_pca_b = model.compress(Xb)
            q_xyz   = model.readout(z_pca_b)
            x_hat   = model.decoder(q_xyz)

            l_con   = contrastive_loss(q_xyz, yb)
            l_recon = reconstruction_loss(x_hat, z_pca_b)
            loss    = LAMBDA_1 * l_con + l_recon

            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            con_list.append(l_con.item())
            recon_list.append(l_recon.item())

        epoch_loss = float(np.mean(loss_list))

        # Relative min_delta: an epoch counts as an improvement only if it beats
        # best_loss by more than REL_MIN_DELTA * best_loss. best_loss=inf on
        # epoch 1 -> always improves.
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
              f'con={np.mean(con_list):.5f}  '
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
# EVALUATION (verbatim from M0a_CADE_original.py)
# ======================================================================

def compute_class_centroids(z_train, y_train):
    classes   = np.unique(y_train)
    centroids = {c: z_train[y_train == c].mean(axis=0) for c in classes}
    return centroids


def compute_mad_scores(z_train, y_train, z_test, centroids):
    """
    Restored: both raw (threshold-free) and MAD-normalized (fixed-
    threshold) distance scores -- same protocol as Group15/M0a.
    """
    classes = list(centroids.keys())
    class_stats = {}
    for c in classes:
        z_c   = z_train[y_train == c]
        dists = np.linalg.norm(z_c - centroids[c], axis=1)
        med   = np.median(dists)
        mad   = 1.4826 * np.median(np.abs(dists - med))
        mad   = max(mad, 1e-6)
        class_stats[c] = {'median': med, 'mad': mad}

    raw_scores, mad_scores = [], []
    for z in z_test:
        dis_k  = [float(np.linalg.norm(z - centroids[c])) for c in classes]
        anom_k = [abs(dis_k[i] - class_stats[c]['median']) / class_stats[c]['mad']
                  for i, c in enumerate(classes)]
        raw_scores.append(float(np.min(dis_k)))
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
        'model':          'Group14_VQC_M1aFix_XYZ_4class',
        'reference':      'M1a-fix: VQC (5 qubits, 20 layers, 200 params) trained and '
                           'scored on the identical 15-dim <X,Y,Z>-per-qubit readout, '
                           'via CADE\'s own exact Euclidean contrastive loss',
        'architecture':   f'{n_features}->PCA({PCA_DIM})->VQC({N_QUBITS}q,{N_LAYERS}L)->'
                           f'readout({READOUT_DIM})  ({N_QUANTUM_PARAMS} quantum trainable params)',
        'loss':           f'{LAMBDA_1}*contrastive(Euclidean,margin={MARGIN}) + reconstruction',
        'run_id':         int(run_id),
        'split_id':       int(split_id),
        'ood_family':     ood_family_name,
        'seed':           int(seed),
        'n_train':        int(n_train),
        'n_test':         int(n_test),
        'n_features':     int(n_features),
        'n_classes':      N_CLASSES,
        'readout_dim':    READOUT_DIM,
        'n_qubits':       N_QUBITS,
        'n_layers':       N_LAYERS,
        'n_quantum_params': N_QUANTUM_PARAMS,
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
    print(f'Group14 VQC M1a-fix (XYZ, 4-class) | Task={task_id} | Run={run_id} | '
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

    print(f'\n[STAGE 1] Training VQC (M1a-fix, XYZ readout)', flush=True)
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

    print(f'\n[STAGE 2] Extracting XYZ readout', flush=True)
    q_train = model.get_readout_numpy(X_train)
    q_test  = model.get_readout_numpy(X_test)
    print(f'  q_train={q_train.shape}  q_test={q_test.shape}', flush=True)

    print(f'\n[STAGE 3] OOD scoring (CADE MAD + threshold-free, on the readout)', flush=True)
    centroids = compute_class_centroids(q_train, y_train)
    raw_scores, mad_scores = compute_mad_scores(q_train, y_train, q_test, centroids)

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
    print('Group14 VQC M1a-FIX (XYZ, 4-class) — AGGREGATED RESULTS')
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
        'model': 'Group14_VQC_M1aFix_XYZ_4class',
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
        description='Group14: VQC M1a-fix (5q, 20L, XYZ readout, CADE loss) — 4-class LAMDA')
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
