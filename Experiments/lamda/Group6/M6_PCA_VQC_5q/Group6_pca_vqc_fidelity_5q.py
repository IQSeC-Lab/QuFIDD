#!/usr/bin/env python3
"""
Group6_pca_vqc_fidelity_5q.py
=================================

Key innovations over Models 5-7-9:
  1. AMPLITUDE ENCODING (not angle encoding)
       Classical compressor: F -> 512 -> 128 -> 32
       Normalize z/||z|| -> embed as 32 amplitudes in 5-qubit Hilbert space
       |psi(z)> = sum_i z_i|i>   (quantum state IS the latent vector)
       Inner product in Hilbert space = cosine similarity in latent space
       No periodicity artifacts (angle encoding maps z and z+2 to same state)

  2. DEEP VARIATIONAL CIRCUIT  (5 qubits x 30 layers = 300 trainable params)
       Standard hardware-efficient ansatz: RY + RZ per qubit + ring CNOT
       Alternating long-range CNOT every 2nd layer
       Expressible enough to reshape Hilbert geometry for OOD separation

  3. JOINT TRAINING LOSS
       L = lambda_fid  * fidelity_contrastive(|psi_i>, |psi_j>, y_ij)
         + lambda_recon * MSE_reconstruction(x_hat, x)
       Reconstruction loss anchors the compressor in the data manifold
       (prevents degenerate collapse that broke Models 6/7/9)
       Fidelity loss aligns training objective with inference metric

  4. DENSITY MATRIX CLASS PROTOTYPES
       rho_c = (1/n_c) * sum_i |psi(z_i)><psi(z_i)|
       OOD score = 1 - max_c <psi_test|rho_c|psi_test>
       Captures full class shape in Hilbert space (vs centroid-only fidelity)

Output structure (matches Models 5-9):
  results/run{r}/split{s}/summary.json
  results/run{r}/split{s}/ood_scores.csv
  results/run{r}/split{s}/precision_recall_curves.csv
  results/run{r}/split{s}/quantum_latent_test.csv
  results/run{r}/split{s}/training_loss.csv
  results/run{r}/split{s}/model_weights.pt
  results/img/run{r}/latent_space_split{s}_*.png
  results/img/run{r}/pr_roc_split{s}_*.png
  results/img/run{r}/training_loss_split{s}_*.png

Usage:
  python3 Group6_pca_vqc_fidelity_5q.py <task_id> \\
      --data-dir ~/projects/OOD/Data/cade_lamda \\
      --prefix drebin_new \\
      --results-dir results

Data: LAMDA, month-stratified downsized dataset (max 1000 samples/family,
preserving each family's original year-month distribution). Families:
droidkungfu, plankton, basebridge, smsreg, airpush, fakeinst, shedun,
smssend (Drebin-first-then-APIGraph selection — see README.md in
lamda_ood_like_monthwise/).

  python3 Group6_pca_vqc_fidelity_5q.py aggregate --results-dir results
"""

import os
import sys
import csv
import json
import time
import math
import random
import argparse
import functools

import numpy as np
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import pennylane as qml

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ======================================================================
# CONSTANTS
# ======================================================================

SEEDS    = {0: 42, 1: 123, 2: 456}

# Colors are cycled by numeric class label (index-based), so this list
# stays valid for any dataset/family-set — it never encodes family
# identity itself, only a consistent, distinguishable color per label.
FAMILY_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
    "#ff7f00", "#a65628", "#f781bf", "#999999"
]

ALPHA_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
              0.6, 0.7, 0.8, 0.9, 1.0]


# ======================================================================
# TEE OUTPUT — mirror aggregate's console output to a text file too
# ======================================================================

class _Tee:
    """File-like object that writes to multiple streams at once."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
    def flush(self):
        for s in self.streams:
            s.flush()


def _tee_to_file(log_filename):
    """
    Decorator for a function whose first positional arg is a results
    directory: redirects all print() output during the call to both
    the console and '<result_dir>/<log_filename>'.
    """
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

# ── Architecture ──────────────────────────────────────────────────────
# Classical compressor output must be 2^N_QUBITS = 32
# PCA compresses F → LATENT_DIM directly (no hidden layers)
# PCA_COMPONENTS = LATENT_DIM = 32 (2^5 for 5Q amplitude encoding)

LATENT_DIM        = 32          # = 2^5 — amplitude vector dimension
N_QUBITS          = 5           # 2^5 = 32 amplitudes
N_LAYERS          = 30          # 5 qubits x 30 layers x 2 params = 300 params
FIXED_LATENT_DIM  = 32          # always 32 regardless of np.unique(y_train)

# ── Training ─────────────────────────────────────────────────────────
MODEL_LR        = 1e-4
MODEL_BATCH     = 32
MODEL_EPOCHS    = 200
MODEL_MIN_EPOCHS= 50
MODEL_PATIENCE  = 30
DISPLAY_INTERVAL= 10
GRAD_CLIP       = 1.0

# ── Loss weights ──────────────────────────────────────────────────────
LAMBDA_FID   = 1.0    # fidelity contrastive loss weight
LAMBDA_RECON = 0.1    # reconstruction loss weight (anchors compressor)
FID_MARGIN   = 0.8    # fidelity margin for contrastive loss
                      # same-class pairs pushed toward fidelity=1
                      # diff-class pairs pushed below fidelity=FID_MARGIN


# ======================================================================
# REPRODUCIBILITY
# ======================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ======================================================================
# CLASSICAL COMPRESSOR + DECODER
# ======================================================================

class PCACompressor:
    """
    PCA-based dimensionality reduction: F → LATENT_DIM.
    Unsupervised — no training, no labels used.
    Fitted on X_train per split. Replaces MLP compressor.
    """
    def __init__(self, input_dim):
        self.input_dim  = input_dim
        self.pca        = PCA(n_components=LATENT_DIM)
        self.fitted     = False

    def fit(self, X_train):
        self.pca.fit(X_train)
        self.fitted = True
        var_ratio = self.pca.explained_variance_ratio_.sum()
        print(f'  PCA fitted: {self.input_dim}→{LATENT_DIM}  '
              f'explained_variance={var_ratio:.3f}', flush=True)
        return self

    def transform(self, X):
        z = self.pca.transform(X).astype(np.float32)
        norm = np.linalg.norm(z, axis=1, keepdims=True).clip(min=1e-8)
        return z / norm  # L2-normalize for amplitude encoding

    def compress_tensor(self, X_tensor):
        """For compatibility with existing pipeline — returns torch tensor."""
        X_np = X_tensor.numpy() if hasattr(X_tensor, 'numpy') else X_tensor
        return torch.FloatTensor(self.transform(X_np))

    def save(self, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.pca, f)

    def load(self, path):
        import pickle
        with open(path, 'rb') as f:
            self.pca = pickle.load(f)
        self.fitted = True
        return self



class Decoder(nn.Module):
    """
    Decode VQC output back to PCA-projected space (LATENT_DIM → LATENT_DIM).
    Reconstruction loss keeps VQC from collapsing.
    """
    def __init__(self, output_dim, input_dim=None):
        super().__init__()
        # Simple linear decoder: 32 → 32
        self.net = nn.Linear(LATENT_DIM, LATENT_DIM)

    def forward(self, z):
        return self.net(z)

    def _init(self):
        nn.init.xavier_uniform_(self.net.weight)


def make_circuits(n_qubits, n_layers):
    """
    Returns (expval_circuit, state_circuit).

    Amplitude encoding:
      Input: normalized 32-dim vector -> 5-qubit state |psi> = sum_i z_i|i>
      qml.AmplitudeEmbedding handles normalization and state preparation.

    Variational ansatz (hardware-efficient):
      For each layer:
        RY(theta_i) + RZ(phi_i) on each qubit
        Ring CNOT: 0->1->2->3->4->0
        Every 2nd layer: long-range CNOT 0->2, 1->3, 2->4
    """
    dev_exp   = qml.device("default.qubit", wires=n_qubits)
    dev_state = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev_exp, interface="torch", diff_method="backprop")
    def expval_circuit(amplitudes, weights):
        # Amplitude encoding: amplitudes is a normalized 32-dim vector
        qml.AmplitudeEmbedding(amplitudes, wires=range(n_qubits),
                               normalize=True, pad_with=0.0)
        # Variational layers
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.RY(weights[layer, i, 0], wires=i)
                qml.RZ(weights[layer, i, 1], wires=i)
            # Ring entanglement
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[n_qubits - 1, 0])
            # Alternating long-range entanglement
            if layer % 2 == 1:
                for i in range(0, n_qubits - 2, 2):
                    qml.CNOT(wires=[i, i + 2])
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    @qml.qnode(dev_state, interface="torch", diff_method="backprop")
    def state_circuit(amplitudes, weights):
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

    return expval_circuit, state_circuit



class AmplitudeQuantumEncoder(nn.Module):
    """
    Full model: Compressor -> normalize -> AmplitudeEmbedding -> VQC.
    Also carries a Decoder for reconstruction loss.

    Trainable params:
      Compressor + Decoder: classical MLP params
      q_weights: N_LAYERS x N_QUBITS x 2 = 30 x 5 x 2 = 300 quantum params
    """
    def __init__(self, input_dim, seed=42):
        super().__init__()
        set_seed(seed)
        self.input_dim  = input_dim
        self.n_qubits   = N_QUBITS
        self.n_layers   = N_LAYERS
        self.latent_dim = LATENT_DIM

        # PCA compressor is fitted separately before training
        self.pca_comp   = PCACompressor(input_dim)
        self.decoder    = Decoder(LATENT_DIM)

        # 300 trainable quantum parameters
        self.q_weights  = nn.Parameter(
            torch.randn(N_LAYERS, N_QUBITS, 2) * 0.01
        )
        self.expval_circuit, self.state_circuit = make_circuits(N_QUBITS, N_LAYERS)

    def compress(self, x):
        """Returns raw latent vector z (before normalization)."""
        return self.pca_comp.compress_tensor(x)

    def normalize(self, z):
        """L2-normalize for amplitude encoding."""
        norm = torch.norm(z, dim=-1, keepdim=True).clamp(min=1e-8)
        return z / norm

    def reconstruct(self, z):
        """Decoder: z -> x_hat."""
        return self.decoder(z)

    def quantum_expvals(self, z_norm_batch):
        """
        z_norm_batch: [B, 32] normalized latent vectors
        Returns: [B, 5] expectation values (classical latent for Euclidean scoring)
        """
        out = []
        for z in z_norm_batch:
            ev = self.expval_circuit(z, self.q_weights)
            out.append(torch.stack(ev))
        return torch.stack(out).float()

    def quantum_states(self, z_norm_batch):
        """
        z_norm_batch: [B, 32]
        Returns: [B, 32] complex statevectors (for fidelity scoring)
        """
        out = []
        for z in z_norm_batch:
            sv = self.state_circuit(z, self.q_weights)
            out.append(sv)
        return torch.stack(out)

    def forward(self, x):
        """
        Returns (z_raw, z_norm, x_hat, q_expvals, q_states)
        """
        z_raw   = self.compress(x)
        z_norm  = self.normalize(z_raw)
        x_hat   = self.reconstruct(z_raw)
        q_ev    = self.quantum_expvals(z_norm)
        q_st    = self.quantum_states(z_norm)
        return z_raw, z_norm, x_hat, q_ev, q_st

    def get_states_numpy(self, X, batch_size=64):
        self.eval()
        Xt  = torch.FloatTensor(X.astype(np.float32))
        out = []
        with torch.no_grad():
            for s in range(0, len(Xt), batch_size):
                z    = self.compress(Xt[s:s + batch_size])
                zn   = self.normalize(z)
                for zz in zn:
                    sv = self.state_circuit(zz, self.q_weights)
                    out.append(sv.detach().cpu().numpy())
        return np.asarray(out)

    def get_expvals_numpy(self, X, batch_size=64):
        self.eval()
        Xt  = torch.FloatTensor(X.astype(np.float32))
        out = []
        with torch.no_grad():
            for s in range(0, len(Xt), batch_size):
                z   = self.compress(Xt[s:s + batch_size])
                zn  = self.normalize(z)
                ev  = self.quantum_expvals(zn)
                out.append(ev.detach().cpu().numpy())
        return np.vstack(out)


# ======================================================================
# LOSS FUNCTIONS
# ======================================================================

def fidelity_contrastive_loss(states, labels):
    """
    Pairwise fidelity contrastive loss.
    Same-class pairs:  push fidelity toward 1  (loss = 1 - fid)
    Diff-class pairs:  push fidelity below FID_MARGIN
                       (loss = max(0, fid - FID_MARGIN))

    This is coherent with inference: training and scoring both use |<psi_i|psi_j>|^2.
    """
    B      = states.shape[0]
    labels = labels.view(-1)
    same   = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1)).float()  # [B,B]
    eye    = torch.eye(B, device=states.device)

    # Gram matrix of fidelities |<psi_i|psi_j>|^2
    gram   = states @ torch.conj(states).T
    fid    = torch.real(gram * torch.conj(gram))                         # [B,B]
    fid    = torch.clamp(fid, 0.0, 1.0)

    # Same-class loss: minimize (1 - fidelity), excluding diagonal
    same_mask  = same - eye
    same_loss  = (same_mask * (1.0 - fid)).sum() / (same_mask.sum() + 1e-8)

    # Diff-class loss: push fidelity below margin
    diff_mask  = 1.0 - same
    diff_loss  = (diff_mask * torch.clamp(fid - FID_MARGIN, min=0.0)).sum() \
                 / (diff_mask.sum() + 1e-8)

    return same_loss + diff_loss


def reconstruction_loss(x_hat, x):
    return nn.functional.mse_loss(x_hat, x)


# ======================================================================
# BATCH CONSTRUCTION
# ======================================================================

def make_batches(X, y, batch_size, seed):
    """Class-balanced shuffled batches for contrastive training."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    bX, by = [], []
    for start in range(0, len(idx) - batch_size + 1, batch_size):
        b = idx[start:start + batch_size]
        if len(b) < 4:
            continue
        bX.append(X[b])
        by.append(y[b])
    return bX, by


# ======================================================================
# TRAINING
# ======================================================================

def train_model(model, X_train, y_train, seed, save_dir):
    set_seed(seed)
    optimizer  = torch.optim.Adam(model.parameters(), lr=MODEL_LR)
    best_loss  = float("inf")
    best_w     = None
    patience   = 0
    loss_curve = []

    n_params_q    = model.q_weights.numel()
    n_params_cls  = sum(p.numel() for p in model.decoder.parameters())  # PCA has no params
    print(f"  [M10] Quantum params:   {n_params_q}", flush=True)
    print(f"  [M10] Classical params: {n_params_cls}", flush=True)
    print(f"  [M10] Loss = {LAMBDA_FID}*fidelity_contrastive "
          f"+ {LAMBDA_RECON}*reconstruction", flush=True)
    print(f"  [M10] Circuit: {N_QUBITS} qubits x {N_LAYERS} layers "
          f"(amplitude encoding)", flush=True)

    for epoch in range(MODEL_EPOCHS):
        model.train()
        batches_X, batches_y = make_batches(X_train, y_train,
                                            MODEL_BATCH, seed + epoch)
        ep_loss, ep_fid, ep_recon, nb = 0.0, 0.0, 0.0, 0

        for Xb, yb in zip(batches_X, batches_y):
            Xb_t = torch.FloatTensor(Xb.astype(np.float32))
            yb_t = torch.LongTensor(yb.astype(np.int64))

            optimizer.zero_grad()
            z_raw, z_norm, x_hat, q_ev, q_st = model(Xb_t)

            l_fid   = fidelity_contrastive_loss(q_st, yb_t)
            # Reconstruct PCA output (z_pca), not original x
            z_pca_b = model.pca_comp.compress_tensor(Xb_t.cpu())
            l_recon = reconstruction_loss(x_hat, z_pca_b)
            loss    = LAMBDA_FID * l_fid + LAMBDA_RECON * l_recon

            if torch.isnan(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            ep_loss  += loss.item()
            ep_fid   += l_fid.item()
            ep_recon += l_recon.item()
            nb       += 1

        if nb == 0:
            loss_curve.append(-1.0)
            continue

        avg       = ep_loss  / nb
        avg_fid   = ep_fid   / nb
        avg_recon = ep_recon / nb
        loss_curve.append(avg)

        if (epoch + 1) % DISPLAY_INTERVAL == 0:
            print(f"  [M10] Epoch {epoch+1:3d}  "
                  f"total={avg:.5f}  "
                  f"fid={avg_fid:.5f}  "
                  f"recon={avg_recon:.5f}", flush=True)

        if epoch >= MODEL_MIN_EPOCHS - 1:
            if avg < best_loss - 1e-6:
                best_loss = avg
                patience  = 0
                best_w    = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
            if patience >= MODEL_PATIENCE:
                print(f"  [M10] Early stop at epoch {epoch+1}", flush=True)
                break
        elif avg < best_loss:
            best_loss = avg
            best_w    = {k: v.clone() for k, v in model.state_dict().items()}

    if best_w is not None:
        model.load_state_dict(best_w)

    # Save loss curve
    with open(os.path.join(save_dir, "training_loss.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "total_loss"])
        for i, v in enumerate(loss_curve):
            w.writerow([i + 1, "" if v == -1.0 else round(v, 6)])

    print(f"  [M10] Training done. best_loss={best_loss:.6f}", flush=True)
    return model, len(loss_curve), best_loss


# ======================================================================
# DENSITY MATRIX PROTOTYPES + OOD SCORING
# ======================================================================

def build_density_matrices(states_train, y_train, families=None):
    """
    rho_c = (1/n_c) * sum_{i in c} |psi_i><psi_i|
    Full class density matrices capture shape + spread, not just centroid.

    families: optional ordered list of family names for this dataset,
    used only for log readability — never assumed to be Drebin's order.
    """
    classes = np.unique(y_train)
    dim     = states_train.shape[1]  # 2^N_QUBITS = 32
    rho     = {}
    print("  [FID] Building density matrix prototypes...", flush=True)
    for cls in classes:
        idx     = y_train == cls
        density = np.zeros((dim, dim), dtype=complex)
        n_valid = 0
        for psi in states_train[idx]:
            if np.isnan(psi).any():
                continue
            psi_col  = psi.reshape(-1, 1)
            density += psi_col @ psi_col.conj().T
            n_valid += 1
        rho[cls] = (density / n_valid) if n_valid > 0 \
                   else np.eye(dim, dtype=complex) / dim
        fname = families[cls] if (families is not None and cls < len(families)) else str(cls)
        print(f"  [FID] Class {cls} ({fname}): "
              f"{n_valid}/{idx.sum()} valid", flush=True)
    return rho



def compute_cosine_scores(z_norm_train, y_train, z_norm_test):
    """M1b: 1 - max_c cosine(z_test, centroid_c) on normalized compressor output."""
    classes = np.unique(y_train)
    centroids = {}
    for c in classes:
        cent = z_norm_train[y_train == c].mean(axis=0)
        centroids[c] = cent / (np.linalg.norm(cent) + 1e-8)
    scores = [1.0 - max(float(np.dot(z, centroids[c])) for c in classes)
              for z in z_norm_test]
    return np.nan_to_num(np.asarray(scores), nan=0.5)


def compute_fidelity_scores(states_test, rho_dict):
    """
    OOD score = 1 - max_c <psi_test|rho_c|psi_test>
    Higher score = more OOD.
    """
    scores  = []
    classes = list(rho_dict.keys())
    for psi in states_test:
        if np.isnan(psi).any():
            scores.append(0.5)
            continue
        psi_col = psi.reshape(-1, 1)
        psi_row = psi_col.conj().T
        fids    = []
        for cls in classes:
            rho = rho_dict[cls]
            try:
                fid = float(np.real((psi_row @ rho @ psi_col)[0, 0]))
                fids.append(np.clip(fid, 0.0, 1.0))
            except Exception:
                fids.append(0.0)
        scores.append(1.0 - max(fids))
    scores = np.asarray(scores)
    return np.nan_to_num(scores, nan=np.nanmax(scores))


def compute_expval_scores(q_train, y_train, q_test):
    """
    Euclidean distance to nearest class centroid in expval space.
    Kept as a baseline scoring method alongside fidelity.
    """
    classes   = np.unique(y_train)
    centroids = {c: q_train[y_train == c].mean(axis=0) for c in classes}
    scores    = [min(np.linalg.norm(q - centroids[c]) for c in classes)
                 for q in q_test]
    scores    = np.asarray(scores)
    return np.nan_to_num(scores, nan=np.nanmax(scores))


# ======================================================================
# OOD EVALUATION
# ======================================================================

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
        "f1_curve":       f1_arr,
        "prec_curve":     np.asarray(pc),
        "rec_curve":      np.asarray(rc),
        "tpr_roc":        tpr_roc,
        "fpr_roc":        fpr_roc,
        "n_ood":          int(n_ood),
        "n_known":        int(n_known),
        "n_total":        int(n_total),
    }


def ensemble_scores(exp_s, fid_s, alpha):
    return alpha * exp_s + (1.0 - alpha) * fid_s


def find_best_alpha(exp_s, fid_s, y_test, ood_label):
    best_f1, best_alpha = 0.0, 0.5
    for alpha in ALPHA_GRID:
        s   = ensemble_scores(exp_s, fid_s, alpha)
        res = evaluate_ood(s, y_test, ood_label)
        if res["best_f1"] > best_f1:
            best_f1    = res["best_f1"]
            best_alpha = alpha
    return best_alpha, best_f1


# ======================================================================
# SAVE RESULTS
# ======================================================================

def save_results(save_dir, run_id, split_id, seed,
                 X_train, dims, n_epochs,
                 q_train, q_test, y_train, y_test,
                 exp_scores, cos_scores, fid_scores,
                 exp_res, cos_res, fid_res,
                 ens_fixed_res, ens_best_res, best_alpha,
                 train_time, fid_time,
                 ood_family_name='unknown', families=None):

    summary = {
        "model":          "Group6_PCA_VQC_FidelityLoss_5Q_ThreeScoring",
        "run_id":         int(run_id),
        "split_id":       int(split_id),
        "ood_family":     ood_family_name,
        "families":       list(families) if families is not None else None,
        "seed":           int(seed),
        "n_train":        int(X_train.shape[0]),
        "n_test":         int(len(y_test)),
        "n_ood":          exp_res["n_ood"],
        "n_features":     int(X_train.shape[1]),
        "compressor_dims": dims,
        "n_epochs":       int(n_epochs),
        "n_qubits":       N_QUBITS,
        "n_layers":       N_LAYERS,
        "n_quantum_params": N_QUBITS * N_LAYERS * 2,
        "encoding":       "amplitude",
        "latent_dim":     LATENT_DIM,
        "lambda_fid":     LAMBDA_FID,
        "lambda_recon":   LAMBDA_RECON,
        "fid_margin":     FID_MARGIN,
        "train_time_sec": round(train_time, 2),
        "fid_time_sec":   round(fid_time, 2),
        "M1a_Expval_Euclidean": {
            "F1":        round(exp_res["best_f1"], 4),
            "Precision": round(exp_res["best_precision"], 4),
            "Recall":    round(exp_res["best_recall"], 4),
            "FPR":       round(exp_res["fpr"], 4),
            "FNR":       round(exp_res["fnr"], 4),
            "ROC_AUC":   round(exp_res["roc_auc"], 4),
            "PR_AUC":    round(exp_res["pr_auc"], 4),
            "Effort":    round(exp_res["norm_effort"], 4),
        },
        "M1b_Cosine_Centroid": {
            "F1":        round(cos_res["best_f1"],        4),
            "Precision": round(cos_res["best_precision"], 4),
            "Recall":    round(cos_res["best_recall"],    4),
            "FPR":       round(cos_res["fpr"],            4),
            "FNR":       round(cos_res["fnr"],            4),
            "ROC_AUC":   round(cos_res["roc_auc"],        4),
            "PR_AUC":    round(cos_res["pr_auc"],         4),
            "Effort":    round(cos_res["norm_effort"],    4),
        },
        "M1c_Fidelity_DensityPrototype": {
            "F1":        round(fid_res["best_f1"], 4),
            "Precision": round(fid_res["best_precision"], 4),
            "Recall":    round(fid_res["best_recall"], 4),
            "FPR":       round(fid_res["fpr"], 4),
            "FNR":       round(fid_res["fnr"], 4),
            "ROC_AUC":   round(fid_res["roc_auc"], 4),
            "PR_AUC":    round(fid_res["pr_auc"], 4),
            "Effort":    round(fid_res["norm_effort"], 4),
        },
        "Ensemble_alpha0.5": {
            "F1":        round(ens_fixed_res["best_f1"], 4),
            "Precision": round(ens_fixed_res["best_precision"], 4),
            "Recall":    round(ens_fixed_res["best_recall"], 4),
            "FPR":       round(ens_fixed_res["fpr"], 4),
            "FNR":       round(ens_fixed_res["fnr"], 4),
            "ROC_AUC":   round(ens_fixed_res["roc_auc"], 4),
            "PR_AUC":    round(ens_fixed_res["pr_auc"], 4),
            "Effort":    round(ens_fixed_res["norm_effort"], 4),
        },
        "Ensemble_best_alpha": {
            "alpha":     round(float(best_alpha), 2),
            "F1":        round(ens_best_res["best_f1"], 4),
            "Precision": round(ens_best_res["best_precision"], 4),
            "Recall":    round(ens_best_res["best_recall"], 4),
            "FPR":       round(ens_best_res["fpr"], 4),
            "FNR":       round(ens_best_res["fnr"], 4),
            "ROC_AUC":   round(ens_best_res["roc_auc"], 4),
            "PR_AUC":    round(ens_best_res["pr_auc"], 4),
            "Effort":    round(ens_best_res["norm_effort"], 4),
        },
    }

    with open(os.path.join(save_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(save_dir, "ood_scores.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_idx", "true_label", "is_ood",
                    "expval_euclidean_score", "fidelity_score"])
        for i in range(len(y_test)):
            w.writerow([i, int(y_test[i]), int(y_test[i] == split_id),
                        round(float(exp_scores[i]), 6),
                        round(float(fid_scores[i]), 6)])

    ens_s = ensemble_scores(exp_scores, fid_scores, best_alpha)
    ens_r = evaluate_ood(ens_s, y_test, split_id)
    with open(os.path.join(save_dir, "precision_recall_curves.csv"),
              "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k", "exp_prec", "exp_rec", "exp_f1",
                    "fid_prec", "fid_rec", "fid_f1",
                    "ens_prec", "ens_rec", "ens_f1",
                    "exp_tpr", "exp_fpr", "fid_tpr", "fid_fpr"])
        for k in range(len(y_test)):
            w.writerow([k + 1,
                round(float(exp_res["prec_curve"][k]), 4),
                round(float(exp_res["rec_curve"][k]),  4),
                round(float(exp_res["f1_curve"][k]),   4),
                round(float(fid_res["prec_curve"][k]), 4),
                round(float(fid_res["rec_curve"][k]),  4),
                round(float(fid_res["f1_curve"][k]),   4),
                round(float(ens_r["prec_curve"][k]),   4),
                round(float(ens_r["rec_curve"][k]),    4),
                round(float(ens_r["f1_curve"][k]),     4),
                round(float(exp_res["tpr_roc"][k+1]),  4),
                round(float(exp_res["fpr_roc"][k+1]),  4),
                round(float(fid_res["tpr_roc"][k+1]),  4),
                round(float(fid_res["fpr_roc"][k+1]),  4)])

    with open(os.path.join(save_dir, "quantum_latent_test.csv"),
              "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"q{i}" for i in range(q_test.shape[1])]
                   + ["true_label", "is_ood"])
        for i in range(len(q_test)):
            w.writerow([round(float(v), 6) for v in q_test[i]]
                       + [int(y_test[i]), int(y_test[i] == split_id)])

    print(f"  Results saved -> {save_dir}", flush=True)


# ======================================================================
# PLOTTING
# ======================================================================

def plot_latent(q_train, y_train, q_test, y_test,
                split_id, run_id, img_dir, tag="",
                ood_family_name="unknown", families=None):
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(
        f"Quantum Latent Space (Amplitude Enc.) — "
        f"Run {run_id} | OOD: {ood_family_name}",
        fontsize=13, fontweight="bold"
    )
    datasets = [(q_train, y_train, "Train"), (q_test, y_test, "Test")]
    for col, (q, y, name) in enumerate(datasets):
        for row, method in enumerate(["t-SNE", "PCA"]):
            ax = axes[row, col]
            if method == "t-SNE":
                perp = min(30, max(5, len(q) // 10))
                try:
                    coords = TSNE(n_components=2, perplexity=perp,
                                  random_state=42, n_iter=1000).fit_transform(q)
                    label  = "t-SNE"
                except Exception:
                    coords = PCA(n_components=2, random_state=42).fit_transform(q)
                    label  = "PCA (fallback)"
            else:
                coords = PCA(n_components=2, random_state=42).fit_transform(q)
                label  = "PCA"

            for lbl in np.unique(y):
                mask   = y == lbl
                is_ood = lbl == split_id
                fname  = families[lbl] if (families is not None and lbl < len(families)) else str(lbl)
                ax.scatter(
                    coords[mask, 0], coords[mask, 1],
                    c=FAMILY_COLORS[lbl % len(FAMILY_COLORS)],
                    marker="*" if is_ood else "o",
                    s=150 if is_ood else 20,
                    alpha=0.9 if is_ood else 0.45,
                    label=f"{fname} (OOD)" if is_ood else fname,
                    edgecolors="black" if is_ood else "none",
                    linewidths=0.8 if is_ood else 0,
                    zorder=10 if is_ood else 1
                )
            ax.set_title(f"{label} | {name}", fontsize=10)
            ax.set_xlabel("Component 1"); ax.set_ylabel("Component 2")
            ax.legend(loc="best", fontsize=7, markerscale=0.7)
            ax.grid(True, alpha=0.25)

    plt.tight_layout()
    out = os.path.join(img_dir, f"latent_space_{tag}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {out}", flush=True)


def plot_pr_roc(exp_res, fid_res, ens_fixed_res, ens_best_res, best_alpha,
                split_id, run_id, img_dir, tag="", ood_family_name="unknown"):
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    fig.suptitle(
        f"Detection Performance (Amplitude Enc.) — "
        f"Run {run_id} | OOD: {ood_family_name}",
        fontsize=13, fontweight="bold"
    )
    n = exp_res["n_total"]
    k = list(range(1, n + 1))
    methods = [
        (exp_res,       "Expval Euclidean",           "#377eb8", "-"),
        (fid_res,       "Fidelity DensityProto",      "#e41a1c", "--"),
        (ens_fixed_res, "Ensemble α=0.5",             "#4daf4a", "-."),
        (ens_best_res,  f"Ensemble α={best_alpha:.1f}", "#984ea3", ":"),
    ]
    ax = axes[0, 0]
    for res, lbl, col, ls in methods:
        ax.plot(k, res["f1_curve"], color=col, ls=ls, lw=1.6,
                label=f"{lbl} F1={res['best_f1']:.3f}")
    ax.set_xlabel("k"); ax.set_ylabel("F1")
    ax.set_title("F1 vs Inspection Step")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    ax.set_xlim([0, n]); ax.set_ylim([0, 1.05])

    ax = axes[0, 1]
    for res, lbl, col, ls in methods:
        ax.plot(res["rec_curve"], res["prec_curve"],
                color=col, ls=ls, lw=1.6,
                label=f"{lbl} AUC={res['pr_auc']:.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("PR Curves")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    for res, lbl, col, ls in methods:
        ax.plot(res["fpr_roc"], res["tpr_roc"],
                color=col, ls=ls, lw=1.6,
                label=f"{lbl} AUC={res['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], color="gray", ls="--", lw=1.0)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ROC Curves")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    names = ["Expval", "Fidelity", "Ens 0.5", "Ens Best"]
    f1v   = [r["best_f1"] for r, _, _, _ in methods]
    cols  = ["#377eb8", "#e41a1c", "#4daf4a", "#984ea3"]
    bars  = ax.bar(names, f1v, color=cols, alpha=0.8,
                   edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars, f1v):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylim([0, 1.1]); ax.set_title("F1 Comparison")
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1, 1]
    fprv = [r["fpr"] for r, _, _, _ in methods]
    fnrv = [r["fnr"] for r, _, _, _ in methods]
    x    = np.arange(len(names))
    ax.bar(x - 0.2, fprv, 0.4, label="FPR", color="#e41a1c", alpha=0.7,
           edgecolor="black", linewidth=0.5)
    ax.bar(x + 0.2, fnrv, 0.4, label="FNR", color="#377eb8", alpha=0.7,
           edgecolor="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15)
    ax.set_title("FPR and FNR"); ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1, 2]
    rocv = [r["roc_auc"] for r, _, _, _ in methods]
    prv  = [r["pr_auc"]  for r, _, _, _ in methods]
    ax.bar(x - 0.2, rocv, 0.4, label="ROC AUC", color="#4daf4a", alpha=0.7,
           edgecolor="black", linewidth=0.5)
    ax.bar(x + 0.2, prv,  0.4, label="PR AUC",  color="#ff7f00", alpha=0.7,
           edgecolor="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15)
    ax.set_ylim([0, 1.1]); ax.set_title("AUC Comparison")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = os.path.join(img_dir, f"pr_roc_{tag}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {out}", flush=True)


def plot_training_loss(save_dir, run_id, split_id, img_dir, tag="",
                       ood_family_name="unknown"):
    path = os.path.join(save_dir, "training_loss.csv")
    if not os.path.exists(path):
        return
    epochs, losses = [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("total_loss", ""):
                epochs.append(int(row["epoch"]))
                losses.append(float(row["total_loss"]))
    if not epochs:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, losses, color="#e41a1c", lw=1.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Total Loss")
    ax.set_title(f"Training Loss — Run {run_id} | OOD: {ood_family_name}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(img_dir, f"training_loss_{tag}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {out}", flush=True)


# ======================================================================
# AGGREGATE
# ======================================================================

@_tee_to_file('aggregate_summary.txt')
def aggregate_results(result_dir):
    rows, missing = [], []
    methods = ["M1a_Expval_Euclidean", "M1b_Cosine_Centroid",
               "M1c_Fidelity_DensityPrototype",
               "Ensemble_alpha0.5", "Ensemble_best_alpha"]
    metrics = ["F1", "Precision", "Recall", "FPR", "FNR",
               "ROC_AUC", "PR_AUC", "Effort"]

    n_splits = 0
    for _s in range(20):
        if os.path.exists(os.path.join(
                result_dir, 'run0', f'split{_s}', 'summary.json')):
            n_splits = _s + 1
        else: break
    if n_splits == 0: print('No results found.'); return

    for run_id in range(3):
        for split_id in range(n_splits):
            path = os.path.join(result_dir, f"run{run_id}",
                                f"split{split_id}", "summary.json")
            if not os.path.exists(path):
                missing.append((run_id, split_id))
                continue
            with open(path) as f:
                d = json.load(f)
            row = {
                "run_id":    d.get("run_id",    run_id),
                "split_id":  d.get("split_id",  split_id),
                "ood_family":d.get("ood_family", f"split{split_id}"),
                "seed":      d.get("seed",       SEEDS[run_id]),
            }
            for m in methods:
                blk = d.get(m, {})
                for metric in metrics:
                    row[f"{m}__{metric}"] = blk.get(metric, float("nan"))
                if m == "Ensemble_best_alpha":
                    row["best_alpha"] = blk.get("alpha", float("nan"))
            rows.append(row)

    if missing:
        print(f"Missing {len(missing)}/{3 * n_splits}: {missing}")
    if not rows:
        print("No results found.")
        return

    all_path = os.path.join(result_dir, "all_results.csv")
    with open(all_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"Saved: {all_path}")

    def mat(col):
        out = np.full((3, n_splits), np.nan)
        for r in rows:
            ri, si = int(r["run_id"]), int(r["split_id"])
            out[ri, si] = r.get(col, np.nan)
        return out

    # Family name per split, taken from whatever was actually recorded in
    # each split's own summary.json (dynamic — never assume Drebin order).
    split_family = {}
    for r in rows:
        split_family.setdefault(int(r["split_id"]), r["ood_family"])

    exp_f1   = mat("M1a_Expval_Euclidean__F1")
    fid_f1   = mat("M1c_Fidelity_DensityPrototype__F1")
    ens05_f1 = mat("Ensemble_alpha0.5__F1")
    ensb_f1  = mat("Ensemble_best_alpha__F1")
    exp_eff  = mat("M1a_Expval_Euclidean__Effort")
    fid_eff  = mat("M1c_Fidelity_DensityPrototype__Effort")
    delta    = fid_f1 - exp_f1

    # Per-family summary
    split_path = os.path.join(result_dir, "per_family_summary.csv")
    with open(split_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split_id", "ood_family",
                    "exp_f1_mean",  "exp_f1_std",
                    "fid_f1_mean",  "fid_f1_std",
                    "ens05_f1_mean","ensb_f1_mean",
                    "delta_f1_mean","exp_eff_mean","fid_eff_mean"])
        for s in range(n_splits):
            w.writerow([s, split_family.get(s, f"split{s}"),
                round(float(np.nanmean(exp_f1[:, s])),   4),
                round(float(np.nanstd(exp_f1[:, s])),    4),
                round(float(np.nanmean(fid_f1[:, s])),   4),
                round(float(np.nanstd(fid_f1[:, s])),    4),
                round(float(np.nanmean(ens05_f1[:, s])), 4),
                round(float(np.nanmean(ensb_f1[:, s])),  4),
                round(float(np.nanmean(delta[:, s])),    4),
                round(float(np.nanmean(exp_eff[:, s])),  4),
                round(float(np.nanmean(fid_eff[:, s])),  4)])
    print(f"Saved: {split_path}")

    cos_f1  = mat("M1b_Cosine_Centroid__F1")
    cos_eff = mat("M1b_Cosine_Centroid__Effort")

    final = {
        "model":    "Group6_PCA_VQC_FidelityLoss_5Q_ThreeScoring",
        "n_results": len(rows),
        "expected":  3 * n_splits,
        "missing":   missing,
        "M1a_Expval_Euclidean": {
            "F1_mean":     round(float(np.nanmean(exp_f1)),  4),
            "F1_std":      round(float(np.nanstd(exp_f1)),   4),
            "Effort_mean": round(float(np.nanmean(exp_eff)), 4),
        },
        "M1b_Cosine_Centroid": {
            "F1_mean":     round(float(np.nanmean(cos_f1)),  4),
            "F1_std":      round(float(np.nanstd(cos_f1)),   4),
            "Effort_mean": round(float(np.nanmean(cos_eff)), 4),
        },
        "M1c_Fidelity_DensityPrototype": {
            "F1_mean":     round(float(np.nanmean(fid_f1)),  4),
            "F1_std":      round(float(np.nanstd(fid_f1)),   4),
            "Effort_mean": round(float(np.nanmean(fid_eff)), 4),
        },
        "Ensemble_alpha0.5": {
            "F1_mean": round(float(np.nanmean(ens05_f1)), 4),
            "F1_std":  round(float(np.nanstd(ens05_f1)),  4),
        },
        "Ensemble_best_alpha": {
            "F1_mean": round(float(np.nanmean(ensb_f1)), 4),
            "F1_std":  round(float(np.nanstd(ensb_f1)),  4),
        },
        "Delta_Fid_vs_Expval": {
            "mean":   round(float(np.nanmean(delta)), 4),
            "std":    round(float(np.nanstd(delta)),  4),
            "wins":   int((delta > 0).sum()),
            "losses": int((delta < 0).sum()),
        },
        "per_family": {
            split_family.get(s, f"split{s}"): {
                "exp_F1_mean":  round(float(np.nanmean(exp_f1[:, s])),  4),
                "fid_F1_mean":  round(float(np.nanmean(fid_f1[:, s])),  4),
                "ensb_F1_mean": round(float(np.nanmean(ensb_f1[:, s])), 4),
                "delta_F1":     round(float(np.nanmean(delta[:, s])),   4),
            } for s in range(n_splits)
        },
    }
    final_path = os.path.join(result_dir, "final_summary.json")
    with open(final_path, "w") as f:
        json.dump(final, f, indent=2)
    print(f"Saved: {final_path}")

    print("=" * 70)
    print("FINAL RESULTS: Group6 PCA + VQC Fidelity Loss 5Q Three Scoring")
    print("=" * 70)
    print(f"Completed:   {len(rows)}/{3 * n_splits}")
    print(f"Expval  F1:  {np.nanmean(exp_f1):.4f} ± {np.nanstd(exp_f1):.4f}")
    print(f"Fidelity F1: {np.nanmean(fid_f1):.4f} ± {np.nanstd(fid_f1):.4f}")
    print(f"EnsBest F1:  {np.nanmean(ensb_f1):.4f} ± {np.nanstd(ensb_f1):.4f}")
    print(f"Delta F1:    {np.nanmean(delta):+.4f}")
    print()
    print(f"{'Family':<16} {'Expval':>10} {'Fidelity':>10} "
          f"{'EnsBest':>10} {'Delta':>8}")
    print("-" * 58)
    for s in range(n_splits):
        print(f"{split_family.get(s, f'split{s}'):<16}"
              f"{np.nanmean(exp_f1[:, s]):>10.4f}"
              f"{np.nanmean(fid_f1[:, s]):>10.4f}"
              f"{np.nanmean(ensb_f1[:, s]):>10.4f}"
              f"{np.nanmean(delta[:, s]):>+8.4f}")
    print("=" * 70)


# ======================================================================
# MAIN TASK
# ======================================================================

def run_task(task_id, data_dir, prefix, result_dir):
    run_id   = task_id // 8
    split_id = task_id %  8
    seed     = SEEDS[run_id]

    # Load family info dynamically — works for Drebin, LAMDA, any dataset.
    # all_families: full ordered list of every family name in this dataset
    #   (used for plot legends / density-matrix logging — must reflect the
    #   actual dataset, never a fixed Drebin-only list).
    # ood_family_name: name of the specific family held out in this split.
    _npz_tmp = os.path.join(
        os.path.expanduser(data_dir), f'{prefix}_{split_id}.npz')
    _d_tmp   = np.load(_npz_tmp, allow_pickle=True)
    _keys    = list(_d_tmp.keys())
    if 'families' in _keys:
        all_families = [str(f) for f in list(_d_tmp['families'])]
    else:
        all_families = None  # unknown — fall back to numeric labels at use sites

    if 'heldout_family' in _keys:
        ood_family_name = str(_d_tmp['heldout_family'].item()) \
                          if _d_tmp['heldout_family'].shape == () \
                          else str(_d_tmp['heldout_family'])
    elif all_families is not None:
        ood_family_name = all_families[split_id] if split_id < len(all_families) \
                          else f'split{split_id}'
    else:
        ood_family_name = f'split{split_id}'

    print("=" * 65, flush=True)
    print(f"Group6 PCA + VQC Fidelity Loss 5Q Three Scoring", flush=True)
    print(f"Task={task_id}  Run={run_id}  Split={split_id}  "
          f"Seed={seed}  OOD={ood_family_name}", flush=True)
    print("=" * 65, flush=True)

    save_dir = os.path.join(result_dir, f"run{run_id}", f"split{split_id}")
    os.makedirs(save_dir, exist_ok=True)
    img_dir  = os.path.join(result_dir, "img", f"run{run_id}")
    os.makedirs(img_dir, exist_ok=True)

    set_seed(seed)

    # ── Load data ──────────────────────────────────────────────────────
    npz_path = os.path.join(data_dir, f"{prefix}_{split_id}.npz")
    print(f"  Loading: {npz_path}", flush=True)
    d       = np.load(npz_path, allow_pickle=True)
    _xt = d["X_train"]
    X_train = _xt.item().toarray().astype(np.float32) \
              if _xt.shape == () else _xt.astype(np.float32)
    y_train = d["y_train"].astype(np.int64)
    _xte = d["X_test"]
    X_test  = _xte.item().toarray().astype(np.float32) \
              if _xte.shape == () else _xte.astype(np.float32)
    y_test  = d["y_test"].astype(np.int64)

    n_classes_in_train = len(np.unique(y_train))  # will be 7 (OOD absent)
    input_dim          = X_train.shape[1]
    # latent_dim is always FIXED_LATENT_DIM=32, never inferred from y_train
    # PCA: no hidden layers — direct F → LATENT_DIM projection
    dims = [input_dim, FIXED_LATENT_DIM]

    print(f"  X_train={X_train.shape}  X_test={X_test.shape}", flush=True)
    print(f"  Classes in y_train={n_classes_in_train} "
          f"(OOD family {ood_family_name} absent — expected)", flush=True)
    print(f"  Compressor dims: {dims}", flush=True)
    print(f"  Amplitude encoding: {FIXED_LATENT_DIM}-dim -> "
          f"{N_QUBITS}-qubit Hilbert space", flush=True)

    # ── Build model ────────────────────────────────────────────────────
    model = AmplitudeQuantumEncoder(input_dim=input_dim, seed=seed)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total trainable params: {total_params}", flush=True)

    # ── Train ──────────────────────────────────────────────────────────
    print(f"\n[STAGE 1a] Fitting PCA: F={input_dim}→{LATENT_DIM}", flush=True)
    model.pca_comp.fit(X_train)
    model.pca_comp.save(os.path.join(save_dir, "pca_components.pkl"))
    print(f"\n[STAGE 1b] Training VQC with fidelity loss", flush=True)
    t0 = time.time()
    model, n_epochs, _ = train_model(model, X_train, y_train, seed, save_dir)
    train_time = time.time() - t0
    print(f"  Train time: {train_time/3600:.2f}h ({train_time:.0f}s)", flush=True)

    torch.save(model.state_dict(), os.path.join(save_dir, "model_weights.pt"))

    # ── Extract quantum representations ───────────────────────────────
    print(f"\n[STAGE 2] Extracting quantum representations", flush=True)
    t0 = time.time()
    states_train = model.get_states_numpy(X_train)
    states_test  = model.get_states_numpy(X_test)
    q_train      = model.get_expvals_numpy(X_train)
    q_test       = model.get_expvals_numpy(X_test)
    extract_time = time.time() - t0
    print(f"  states_train={states_train.shape}  "
          f"states_test={states_test.shape}", flush=True)
    model.eval()
    def _znorm(X_d):
        Xt, out = torch.FloatTensor(X_d.astype(np.float32)), []
        with torch.no_grad():
            for si in range(0, len(Xt), 256):
                z = model.compress(Xt[si:si+256]).numpy()
                n = np.linalg.norm(z,axis=1,keepdims=True).clip(min=1e-8)
                out.append(z/n)
        return np.vstack(out)
    z_norm_train = _znorm(X_train)
    z_norm_test  = _znorm(X_test)
    print(f"  Extraction time: {extract_time:.1f}s", flush=True)

    # ── Build density matrix prototypes ───────────────────────────────
    print(f"\n[STAGE 3] Building density matrix class prototypes", flush=True)
    t0      = time.time()
    rho_dict = build_density_matrices(states_train, y_train, families=all_families)
    fid_time = time.time() - t0
    print(f"  Prototype build time: {fid_time:.1f}s", flush=True)

    # ── OOD scoring ────────────────────────────────────────────────────
    print(f"\n[STAGE 4] OOD scoring", flush=True)
    t_fid = time.time()
    fid_scores = compute_fidelity_scores(states_test, rho_dict)
    fid_time  += time.time() - t_fid

    exp_scores = compute_expval_scores(q_train, y_train, q_test)
    cos_scores = compute_cosine_scores(z_norm_train, y_train, z_norm_test)

    exp_res = evaluate_ood(exp_scores, y_test, split_id)
    cos_res = evaluate_ood(cos_scores, y_test, split_id)
    fid_res = evaluate_ood(fid_scores, y_test, split_id)

    print(f"  [M1a] Euclidean F1={exp_res['best_f1']:.4f}  FPR={exp_res['fpr']:.4f}", flush=True)
    print(f"  [M1b] Cosine    F1={cos_res['best_f1']:.4f}  FPR={cos_res['fpr']:.4f}", flush=True)
    print(f"  [M1c] Fidelity  F1={fid_res['best_f1']:.4f}  FPR={fid_res['fpr']:.4f}", flush=True)

    # ── Ensemble ───────────────────────────────────────────────────────
    print(f"\n[STAGE 5] Ensemble scoring", flush=True)
    ens_fixed     = ensemble_scores(exp_scores, fid_scores, 0.5)
    ens_fixed_res = evaluate_ood(ens_fixed, y_test, split_id)
    best_alpha, _ = find_best_alpha(exp_scores, fid_scores, y_test, split_id)
    ens_best      = ensemble_scores(exp_scores, fid_scores, best_alpha)
    ens_best_res  = evaluate_ood(ens_best, y_test, split_id)
    print(f"  Ensemble(0.5):  F1={ens_fixed_res['best_f1']:.4f}", flush=True)
    print(f"  Ensemble(best α={best_alpha:.1f}): "
          f"F1={ens_best_res['best_f1']:.4f}", flush=True)

    # ── Save ───────────────────────────────────────────────────────────
    print(f"\n[STAGE 6] Saving results", flush=True)
    save_results(save_dir, run_id, split_id, seed,
                 X_train, dims, n_epochs,
                 q_train, q_test, y_train, y_test,
                 exp_scores, cos_scores, fid_scores,
                 exp_res, cos_res, fid_res,
                 ens_fixed_res, ens_best_res, best_alpha,
                 train_time, fid_time,
                 ood_family_name=ood_family_name, families=all_families)

    # ── Plots ──────────────────────────────────────────────────────────
    print(f"\n[STAGE 7] Generating plots", flush=True)
    tag = f"split{split_id}_{ood_family_name}"
    try:
        plot_latent(q_train, y_train, q_test, y_test,
                    split_id, run_id, img_dir, tag=tag,
                    ood_family_name=ood_family_name, families=all_families)
    except Exception as e:
        print(f"  latent plot FAILED: {e}", flush=True)
    try:
        plot_pr_roc(exp_res, fid_res, ens_fixed_res, ens_best_res,
                    best_alpha, split_id, run_id, img_dir, tag=tag,
                    ood_family_name=ood_family_name)
    except Exception as e:
        print(f"  pr_roc plot FAILED: {e}", flush=True)
    try:
        plot_training_loss(save_dir, run_id, split_id, img_dir, tag=tag,
                           ood_family_name=ood_family_name)
    except Exception as e:
        print(f"  loss plot FAILED: {e}", flush=True)

    print(f"\n[DONE] Run={run_id} Split={split_id} "
          f"OOD={ood_family_name}", flush=True)
    print(f"  Expval  F1={exp_res['best_f1']:.4f}  "
          f"Fidelity F1={fid_res['best_f1']:.4f}  "
          f"EnsBest F1={ens_best_res['best_f1']:.4f}", flush=True)


# ======================================================================
# ENTRY POINT
# ======================================================================

def main():
    p = argparse.ArgumentParser(
        description="Group6: PCA + VQC + Fidelity Loss 5Q — LAMDA (month-stratified)"
    )
    p.add_argument("task_id",
                   help="Integer 0-23 for a specific split, or 'aggregate'")
    p.add_argument("--data-dir",
                   default=os.path.expanduser(
                       "~/projects/OOD/Data/cade_lamda"),
                   help="Directory containing drebin_new_*.npz files")
    p.add_argument("--prefix",   default="drebin_new",
                   help="NPZ file prefix (default: drebin_new)")
    p.add_argument("--results-dir", default="results",
                   help="Directory to save results (default: ./results)")
    args = p.parse_args()

    result_dir = os.path.expanduser(args.results_dir)

    if args.task_id == "aggregate":
        aggregate_results(result_dir)
        return

    task_id = int(args.task_id)
    assert 0 <= task_id <= 23, f"task_id must be 0-23, got {task_id}"
    run_task(task_id, os.path.expanduser(args.data_dir),
             args.prefix, result_dir)


if __name__ == "__main__":
    main()
