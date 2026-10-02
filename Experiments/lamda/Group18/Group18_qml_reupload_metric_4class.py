#!/usr/bin/env python3
"""
Group18_qml_reupload_metric_4class.py
=====================================
Quantum metric learning with DATA RE-UPLOADING -- the quantum counterpart
of CADE -- on the 4-class LAMDA / Drebin leave-one-family-out splits.

WHY THIS DESIGN (what was wrong with Groups 1/6/14/17)
------------------------------------------------------
Groups 14/17 amplitude-encode the normalized PCA vector z ONCE and then
apply a single trainable unitary U(theta). Every measured quantity is then
a quadratic form of z:  <z|U^dag O U|z> = z^T M(theta) z.  Training only
chooses which quadratic features of a fixed 32-dim vector to read out,
which is why those models track PCA and cannot learn structure beyond it.
Scoring by fidelity between states that all passed through the same U is
worse still: it is theta-independent (unitary invariance, Groups 1/6).

Data re-uploading fixes both problems:
  * The input is encoded L times, interleaved with trainable layers:
        |psi(x;theta)> = W_L(theta_L) S(x) ... W_1(theta_1) S(x) |0>
    Each extra upload raises the degree (frequency content) of the
    functions the circuit can represent (Perez-Salinas et al. 2020;
    Schuld, Sweke, Meyer 2021), so the embedding is genuinely nonlinear.
  * The trainable gates sit BETWEEN data encodings, so the embedding
    itself depends on theta and the fidelity |<psi(x)|psi(x')>|^2 is NOT
    theta-invariant. A fidelity-based contrastive loss therefore trains
    the embedding: quantum metric learning (Lloyd, Schuld, Ijaz, Izaac,
    Killoran 2020, "Quantum embeddings for machine learning").

MODEL
-----
  Frozen preprocessing (0 trainable params, fit on the split's training set):
    X -> PCA(N_FEATURES=5) -> center, then divide ALL components by ONE shared
      factor (the std of the first component) -- this keeps PCA's relative
      variance weighting (per-component standardization would inflate the
      weak components to the same scale as the strong ones)
      -> angle a = arctan(z) in (-pi/2, pi/2): monotone, no wrap-around, and
         RY(a)|0> stays on one half of the Bloch sphere, so each qubit's
         encoding is one-to-one
  Circuit, 5 qubits, 20 trainable layers; the data is uploaded N_UPLOADS = 4
  times, before layers 0, 5, 10, 15:
    S(x):   RY(a_k) on qubit k (k=0..4): ONE feature per qubit
    W(th):  RY(th_{l,k,0}), RZ(th_{l,k,1}) on every qubit, CNOT ring,
            plus CNOT 0->2, 2->4 on odd layers (same entangler as Group14/17)
  Trainable parameters: 5 qubits x 20 layers x 2 = 200  (no decoder at all;
  Group15 CADE-mini has 215 including its decoder).

  Why 4 uploads (not 20): with many uploads the embedding scrambles every
  input into a near-random state, all pairwise fidelities concentrate at
  1/32 and there is nothing to learn from (checked at initialization with
  20 full-range uploads: mean fidelity 0.032 for same- AND different-class
  pairs). 4 uploads keep fidelities in a trainable range at initialization.

TRAINING (CADE's objective, on a quantum distance)
--------------------------------------------------
  d(x, x') = sqrt(1 - |<psi(x)|psi(x')>|^2)   in [0, 1]
  CADE contrastive loss with CADE's exact pairing (epoch_batches, 50% same-
  class pairs):  loss = same * d + (1 - same) * relu(MARGIN - d)
  MARGIN = 1.0 (= the maximum possible distance: different classes are
  pushed toward orthogonal states). No reconstruction term / decoder.
  Adam, LR 0.001, max 250 epochs, early stopping (PATIENCE, relative
  min-delta 1e-4, training loss), best-loss weights restored -- same rules
  as Group14/17.

SCORING (theta-dependent, CADE-style)
-------------------------------------
  Class prototype = density matrix rho_c = mean_i |psi_i><psi_i| (training).
  d_c(x) = 1 - <psi(x)|rho_c|psi(x)>  = 1 - mean_i |<psi(x)|psi_i>|^2
  (leave-one-out for training samples). raw = min_c d_c ;
  MAD = min_c |d_c - median_c| / MAD_c (b = 1.4826). Threshold-free metrics
  on raw, MAD threshold 3.5 -- identical output format to Group14/17, so
  cade_protocol_eval.py, pca_ensemble_eval.py, and the ablations work on it.

Usage:
  python3 Group18_qml_reupload_metric_4class.py <task_id 0-11> --data-dir ... --results-dir ...
  python3 Group18_qml_reupload_metric_4class.py <task_id> --max-epochs 30 --results-dir pilot   # pilot
  python3 Group18_qml_reupload_metric_4class.py aggregate --results-dir results
"""

import os, sys, csv, json, time, copy, pickle, argparse, functools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score


# ======================================================================
# TEE (log aggregate output to file)
# ======================================================================
class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, s):
        for st in self.streams:
            st.write(s); st.flush()
    def flush(self):
        for st in self.streams:
            st.flush()


def _tee_to_file(log_filename):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(result_dir, *a, **k):
            path = os.path.join(result_dir, log_filename)
            with open(path, 'w') as f:
                old = sys.stdout
                sys.stdout = _Tee(old, f)
                try:
                    return fn(result_dir, *a, **k)
                finally:
                    sys.stdout = old
        return wrapper
    return deco


# ======================================================================
# CONFIG
# ======================================================================
N_CLASSES       = 4
N_SPLITS        = N_CLASSES
SEEDS           = {0: 42, 1: 123, 2: 456}

N_QUBITS        = 5
N_FEATURES      = N_QUBITS          # 5 PCA components: one RY-encoded feature per qubit
N_LAYERS        = 20                # trainable layers
N_UPLOADS       = 4                 # data uploads (before layers 0, 5, 10, 15)
UPLOAD_EVERY    = N_LAYERS // N_UPLOADS
ANGLE_SCALE     = 1.0               # angle = arctan(z) in (-pi/2, pi/2)
N_QUANTUM_PARAMS = N_QUBITS * N_LAYERS * 2   # = 200
INIT_STD        = 0.1               # trainable-angle init std. Not near-identity: with
                                    # re-uploading, W ~ I would make the embedding just
                                    # the repeated encoding S(x)^L.

MARGIN          = 1.0               # max of d = sqrt(1 - F) is 1 (orthogonal states)
SIMILAR_RATIO   = 0.25              # CADE default -> num_sim = int(64 * 0.25) = 16 of 32
CAE_LR          = 0.001             # same as Group14/15/17
CAE_BATCH_SIZE  = 64                # 32 anchors + 32 partners (CADE pairing)
CAE_EPOCHS      = 250               # ceiling; early stopping below
PATIENCE        = 5
REL_MIN_DELTA   = 1e-4
MAD_THRESHOLD   = 3.5

DEFAULT_DATA_DIR    = os.path.expanduser('~/projects/OOD/Data/cade_lamda_4class_vt')
DEFAULT_RESULTS_DIR = 'results'


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


# ======================================================================
# FROZEN PREPROCESSING: PCA(5) -> shared scale -> angles
# ======================================================================
class AnglePreprocessor:
    """X -> PCA(N_FEATURES) -> center, divide by ONE shared scale (std of PC1) -> ANGLE_SCALE*arctan(z).
    Keeps PCA's relative variance weighting. Frozen after fit."""
    def __init__(self):
        self.pca = PCA(n_components=N_FEATURES, svd_solver='full')
        self.mu = None
        self.sd = None

    def fit(self, X_train):
        Z = self.pca.fit_transform(X_train)
        self.mu = Z.mean(axis=0)
        self.sd = float(Z[:, 0].std()) + 1e-8          # ONE shared scale factor
        ev = float(self.pca.explained_variance_ratio_.sum())
        print(f'  PCA fitted: {X_train.shape[1]}->{N_FEATURES}  explained_variance={ev:.3f}', flush=True)
        return self

    def transform(self, X):
        Z = (self.pca.transform(X) - self.mu) / self.sd
        return (ANGLE_SCALE * np.arctan(Z)).astype(np.float32)


# ======================================================================
# RE-UPLOADING CIRCUIT
# ======================================================================
def make_reupload_circuit(n_qubits, n_layers):
    dev = qml.device('default.qubit', wires=n_qubits)

    @qml.qnode(dev, interface='torch', diff_method='backprop')
    def circuit(angles, weights):
        # angles: [N_QUBITS] or broadcast [B, N_QUBITS]
        for layer in range(n_layers):
            if layer % UPLOAD_EVERY == 0:                  # S(x): data (re-)upload
                for k in range(n_qubits):
                    qml.RY(angles[..., k], wires=k)        # one feature per qubit
            for k in range(n_qubits):                      # W(theta): trainable
                qml.RY(weights[layer, k, 0], wires=k)
                qml.RZ(weights[layer, k, 1], wires=k)
            for k in range(n_qubits - 1):
                qml.CNOT(wires=[k, k + 1])
            qml.CNOT(wires=[n_qubits - 1, 0])
            if layer % 2 == 1:
                for k in range(0, n_qubits - 2, 2):
                    qml.CNOT(wires=[k, k + 2])
        return qml.state()

    return circuit


class ReuploadEmbedding(nn.Module):
    """Frozen angle preprocessing -> re-uploading VQC -> 32-dim complex state. 200 params."""
    def __init__(self, seed=42, broadcast=True):
        super().__init__()
        set_seed(seed)
        self.prep      = AnglePreprocessor()
        self.q_weights = nn.Parameter(torch.randn(N_LAYERS, N_QUBITS, 2) * INIT_STD)
        self.circuit   = make_reupload_circuit(N_QUBITS, N_LAYERS)
        self.broadcast = broadcast

    def states_from_angles(self, A):
        """A: torch [B, N_FEATURES] -> complex [B, 32]."""
        if self.broadcast:
            psi = self.circuit(A, self.q_weights)
            return psi if psi.dim() == 2 else psi.unsqueeze(0)
        return torch.stack([self.circuit(a, self.q_weights) for a in A])

    def states_numpy(self, X, batch_size=128):
        A = torch.from_numpy(self.prep.transform(X))
        out = []
        with torch.no_grad():
            for s in range(0, len(A), batch_size):
                out.append(self.states_from_angles(A[s:s + batch_size]).cpu().numpy())
        return np.vstack(out)


# ======================================================================
# LOSS + CADE PAIRING
# ======================================================================
def quantum_distance(psi_a, psi_b):
    """d = sqrt(1 - |<a|b>|^2), row-wise."""
    ov  = (psi_a.conj() * psi_b).sum(dim=-1)
    fid = (ov.real ** 2 + ov.imag ** 2).clamp(0.0, 1.0)
    return torch.sqrt((1.0 - fid).clamp(min=0.0) + 1e-10)


def contrastive_loss(states, labels):
    """CADE's exact contrastive loss form, on the quantum distance."""
    half = states.shape[0] // 2
    d    = quantum_distance(states[:half], states[half:])
    same = (labels[:half] == labels[half:]).float()
    return (same * d + (1.0 - same) * torch.relu(MARGIN - d)).mean()


def epoch_batches(X_train, y_train, batch_size, similar_ratio):
    """Faithful port of CADE's epoch_batches (same code as Group14/15)."""
    if batch_size % 4 != 0:
        raise ValueError('batch_size should be a multiple of 4.')
    half_size   = batch_size // 2
    n_train     = X_train.shape[0]
    batch_count = int(n_train / half_size)
    num_sim     = int(batch_size * similar_ratio)
    b_out_x = np.zeros([batch_count, batch_size, X_train.shape[1]], dtype=np.float32)
    b_out_y = np.zeros([batch_count, batch_size], dtype=np.int64)
    random_idx = np.random.permutation(n_train)
    b_out_x[:, :half_size] = np.split(X_train[random_idx[: batch_count * half_size]], batch_count)
    b_out_y[:, :half_size] = np.split(y_train[random_idx[: batch_count * half_size]], batch_count)
    classes = np.unique(y_train)
    max_label = int(classes.max())
    index_cls, index_no_cls = [None] * (max_label + 1), [None] * (max_label + 1)
    for label in classes:
        label = int(label)
        index_cls[label]    = np.where(y_train == label)[0]
        index_no_cls[label] = np.where(y_train != label)[0]
    for b in range(batch_count):
        for m in range(0, num_sim):
            pair = np.random.choice(index_cls[int(b_out_y[b, m])], 1)
            b_out_x[b, m + half_size] = X_train[pair[0]]; b_out_y[b, m + half_size] = y_train[pair[0]]
        for m in range(num_sim, half_size):
            pair = np.random.choice(index_no_cls[int(b_out_y[b, m])], 1)
            b_out_x[b, m + half_size] = X_train[pair[0]]; b_out_y[b, m + half_size] = y_train[pair[0]]
    return batch_count, b_out_x, b_out_y


# ======================================================================
# TRAINING
# ======================================================================
def train_model(model, X_train, y_train, seed, max_epochs):
    print(f'\n  [STAGE 1a] Fitting frozen preprocessing', flush=True)
    model.prep.fit(X_train)
    A_train = model.prep.transform(X_train)          # angles, [n, 5]

    optimizer = optim.Adam([model.q_weights], lr=CAE_LR)
    print(f'\n  [STAGE 1b] Training re-uploading embedding (quantum metric learning)', flush=True)
    print(f'  Trainable params: {model.q_weights.numel()} (quantum only; no decoder)', flush=True)
    print(f'  Circuit: {N_QUBITS} qubits x {N_LAYERS} trainable layers, {N_UPLOADS} data uploads '
          f'x {N_FEATURES} features, angle scale {ANGLE_SCALE}', flush=True)
    print(f'  Loss: CADE contrastive on d=sqrt(1-F), margin={MARGIN}, similar_ratio={SIMILAR_RATIO}', flush=True)
    print(f'  LR: {CAE_LR}  Max epochs: {max_epochs}  Patience: {PATIENCE}  RelMinDelta: {REL_MIN_DELTA}', flush=True)

    set_seed(seed)
    best_loss, best_epoch, best_state = float('inf'), 0, copy.deepcopy(model.state_dict())
    since, stopped_early, actual_epochs = 0, False, max_epochs
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        bc, bx, by = epoch_batches(A_train, y_train, CAE_BATCH_SIZE, SIMILAR_RATIO)
        losses, d_same, d_diff = [], [], []
        for b in range(bc):
            A  = torch.from_numpy(bx[b]); yb = torch.from_numpy(by[b])
            optimizer.zero_grad()
            st   = model.states_from_angles(A)
            loss = contrastive_loss(st, yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            with torch.no_grad():
                half = st.shape[0] // 2
                d = quantum_distance(st[:half], st[half:])
                same = (yb[:half] == yb[half:])
                d_same.append(d[same].mean().item()); d_diff.append(d[~same].mean().item())
        epoch_loss = float(np.mean(losses))
        if epoch_loss < best_loss * (1.0 - REL_MIN_DELTA):
            best_loss, best_epoch, since = epoch_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            since += 1
        print(f'  Epoch {epoch:3d}  loss={epoch_loss:.5f}  d_same={np.mean(d_same):.4f}  '
              f'd_diff={np.mean(d_diff):.4f}  gap={np.mean(d_diff) - np.mean(d_same):+.4f}  '
              f'best={best_loss:.5f}  no_improve={since}  ({time.time() - t0:.0f}s)', flush=True)
        if since >= PATIENCE:
            print(f'\n  [EARLY STOP] No improvement for {PATIENCE} epochs '
                  f'(best_loss={best_loss:.5f} at epoch {best_epoch}). Stopping at epoch {epoch}.', flush=True)
            stopped_early, actual_epochs = True, epoch
            break
    model.load_state_dict(best_state)
    print(f'  [RESTORE] Using weights from best epoch {best_epoch} (best_loss={best_loss:.5f}).', flush=True)
    return model, actual_epochs, stopped_early, best_epoch, best_loss


# ======================================================================
# SCORING: density-matrix class prototypes (theta-dependent here)
# ======================================================================
def density_prototype_scores(st_tr, y_tr, st_te):
    classes = sorted(int(c) for c in np.unique(y_tr))
    stats, D = {}, []
    for c in classes:
        Pc = st_tr[y_tr == c]                                  # [n_c, 32]
        n_c = len(Pc)
        F_tr = np.abs(Pc.conj() @ Pc.T) ** 2                   # [n_c, n_c]
        d_tr = 1.0 - (F_tr.sum(axis=1) - 1.0) / (n_c - 1)      # leave-one-out
        med = np.median(d_tr)
        stats[c] = (med, max(1.4826 * np.median(np.abs(d_tr - med)), 1e-6))
        F_te = np.abs(st_te.conj() @ Pc.T) ** 2                # [n_test, n_c]
        D.append(1.0 - F_te.mean(axis=1))
    D = np.stack(D, axis=1)
    A = np.stack([np.abs(D[:, j] - stats[c][0]) / stats[c][1] for j, c in enumerate(classes)], axis=1)
    return D.min(axis=1), A.min(axis=1)


def centroid_baseline(F_tr, y_tr, F_te, is_ood):
    """Zero-parameter classical baseline on the SAME input features: Euclidean distance to the
    nearest known-class centroid. Returns ROC-AUC."""
    D = np.stack([np.linalg.norm(F_te - F_tr[y_tr == c].mean(axis=0), axis=1) for c in np.unique(y_tr)], axis=1)
    return float(roc_auc_score(is_ood, D.min(axis=1)))


def evaluate_ood(scores, y_test, ood_label):
    n_ood = int(np.sum(y_test == ood_label)); n_known = len(y_test) - n_ood; n_total = len(y_test)
    order = np.argsort(scores)[::-1]; y_s = y_test[order]
    tp_c = np.cumsum(y_s == ood_label); k = np.arange(1, n_total + 1)
    pc, rc = tp_c / k, tp_c / n_ood
    f1 = np.where(pc + rc > 0, 2 * pc * rc / (pc + rc + 1e-12), 0.0)
    b = int(np.argmax(f1)); tp_v = int(tp_c[b]); fp_v = (b + 1) - tp_v
    tpr = np.concatenate([[0.0], tp_c / n_ood]); fpr_c = np.concatenate([[0.0], (k - tp_c) / n_known])
    roc_auc = float(np.clip(np.trapezoid(tpr, fpr_c), 0.0, 1.0))
    pr_auc  = float(np.clip(np.trapezoid(np.concatenate([[1.0], pc]), np.concatenate([[0.0], rc])), 0.0, 1.0))
    return {'best_f1': float(f1[b]), 'best_precision': float(pc[b]), 'best_recall': float(rc[b]),
            'norm_effort': float((b + 1) / n_ood), 'fpr': float(fp_v / n_known),
            'fnr': float((n_ood - tp_v) / n_ood), 'roc_auc': roc_auc, 'pr_auc': pr_auc,
            'n_ood': n_ood, 'n_known': n_known, 'n_total': n_total}


def evaluate_mad_threshold(mad_scores, y_test, ood_label, threshold=MAD_THRESHOLD):
    pred = mad_scores > threshold; act = y_test == ood_label
    n_ood = int(act.sum()); n_known = len(y_test) - n_ood
    tp = int((pred & act).sum()); fp = int((pred & ~act).sum()); fn = int((~pred & act).sum())
    tn = int((~pred & ~act).sum())
    p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {'threshold': float(threshold), 'best_f1': round(f1, 4), 'best_precision': round(p, 4),
            'best_recall': round(r, 4), 'fpr': round(fp / n_known, 4) if n_known else 0.0,
            'fnr': round(fn / n_ood, 4) if n_ood else 0.0,
            'norm_effort': round((tp + fp) / n_ood, 4) if n_ood else 0.0,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}


# ======================================================================
# RUN ONE TASK
# ======================================================================
def load_split(data_dir, prefix, s):
    d = np.load(os.path.join(data_dir, f'{prefix}_{s}.npz'), allow_pickle=True)
    def dense(a):
        return a.item().toarray().astype(np.float32) if a.shape == () else a.astype(np.float32)
    hf = d['heldout_family']
    fam = str(hf.item()) if hf.shape == () else str(hf)
    return dense(d['X_train']), d['y_train'].astype(np.int64), dense(d['X_test']), d['y_test'].astype(np.int64), fam


def run_task(task_id, data_dir, prefix, result_dir, max_epochs, broadcast):
    run_id, split_id = task_id // N_SPLITS, task_id % N_SPLITS
    seed = SEEDS[run_id]
    X_tr, y_tr, X_te, y_te, fam = load_split(data_dir, prefix, split_id)
    print('=' * 70, flush=True)
    print(f'Group18 Re-uploading Quantum Metric Learning (4-class) | Task={task_id} | Run={run_id} | '
          f'Split={split_id} | Seed={seed} | OOD={fam}', flush=True)
    print('=' * 70, flush=True)
    print(f'  X_train={X_tr.shape}  X_test={X_te.shape}  classes in y_train={len(np.unique(y_tr))}', flush=True)
    save_dir = os.path.join(result_dir, f'run{run_id}', f'split{split_id}')
    os.makedirs(save_dir, exist_ok=True)

    set_seed(seed)
    model = ReuploadEmbedding(seed=seed, broadcast=broadcast)
    t0 = time.time()
    model, actual_epochs, stopped_early, best_epoch, best_loss = train_model(model, X_tr, y_tr, seed, max_epochs)
    train_time = time.time() - t0
    print(f'  Training time: {train_time:.1f}s (ran {actual_epochs}/{max_epochs} epochs, '
          f'early_stopped={stopped_early}, best_epoch={best_epoch})', flush=True)

    torch.save(model.state_dict(), os.path.join(save_dir, 'model_weights.pt'))
    with open(os.path.join(save_dir, 'preprocessor.pkl'), 'wb') as f:
        pickle.dump(model.prep, f)

    print(f'\n[STAGE 2] Embedding train/test samples', flush=True)
    st_tr = model.states_numpy(X_tr); st_te = model.states_numpy(X_te)
    np.savez_compressed(os.path.join(save_dir, 'states.npz'), states_train=st_tr, states_test=st_te,
                        y_train=y_tr, y_test=y_te)

    print(f'\n[STAGE 3] OOD scoring (density-matrix prototypes; theta-dependent)', flush=True)
    raw, mad = density_prototype_scores(st_tr, y_tr, st_te)
    ood_label = (set(np.unique(y_te).tolist()) - set(np.unique(y_tr).tolist())).pop()
    tf_res  = evaluate_ood(raw, y_te, ood_label)
    mad_res = evaluate_mad_threshold(mad, y_te, ood_label)
    is_ood  = (y_te == ood_label).astype(int)
    roc_mad = float(roc_auc_score(is_ood, mad))
    # Paired zero-parameter baselines on exactly what the circuit sees (PCA-5):
    def l2n(Z):
        return Z / np.linalg.norm(Z, axis=1, keepdims=True).clip(min=1e-8)
    Ztr, Zte = model.prep.pca.transform(X_tr), model.prep.pca.transform(X_te)
    baselines = {'pca5_l2norm_centroid_roc': centroid_baseline(l2n(Ztr), y_tr, l2n(Zte), is_ood),
                 'input_angles_centroid_roc': centroid_baseline(model.prep.transform(X_tr), y_tr,
                                                                model.prep.transform(X_te), is_ood)}
    print(f'  [MAD]  F1={mad_res["best_f1"]:.4f}  flagged={(mad > MAD_THRESHOLD).mean():.3f}', flush=True)
    print(f'  [ThreshFree] F1={tf_res["best_f1"]:.4f}  ROC_AUC={tf_res["roc_auc"]:.4f}  '
          f'PR_AUC={tf_res["pr_auc"]:.4f}  ROC_AUC(MAD score)={roc_mad:.4f}', flush=True)
    print(f'  [Baselines, 0 params, same PCA-5 input] PCA5 centroid ROC='
          f'{baselines["pca5_l2norm_centroid_roc"]:.4f}  input-angle centroid ROC='
          f'{baselines["input_angles_centroid_roc"]:.4f}', flush=True)

    summary = {
        'model': 'Group18_QML_Reupload_Metric_4class',
        'reference': 'Quantum metric learning (Lloyd et al. 2020) with data re-uploading '
                     '(Perez-Salinas et al. 2020); CADE contrastive loss and pairing',
        'architecture': f'{X_tr.shape[1]}->PCA({N_FEATURES})->shared-scale->{ANGLE_SCALE}*arctan -> RY '
                        f'VQC({N_QUBITS}q, {N_LAYERS} layers, {N_UPLOADS} uploads) -> 32-dim state',
        'loss': f'CADE contrastive on d=sqrt(1-fidelity), margin={MARGIN}',
        'run_id': run_id, 'split_id': split_id, 'ood_family': fam, 'seed': seed,
        'n_train': len(y_tr), 'n_test': len(y_te), 'n_features': X_tr.shape[1],
        'n_quantum_params': N_QUANTUM_PARAMS, 'n_trainable_params': N_QUANTUM_PARAMS,
        'lr': CAE_LR, 'max_epochs': max_epochs, 'actual_epochs': actual_epochs,
        'stopped_early': stopped_early, 'best_epoch': best_epoch, 'best_loss': best_loss,
        'patience': PATIENCE, 'rel_min_delta': REL_MIN_DELTA,
        'pca_explained_variance': float(model.prep.pca.explained_variance_ratio_.sum()),
        'train_time_sec': round(train_time, 2), 'roc_auc_mad_score': roc_mad,
        'baselines_same_input': baselines,
        'MAD_Threshold': mad_res, 'Threshold_Free': tf_res,
    }
    with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(save_dir, 'ood_scores.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['raw_score', 'mad_score'])
        for a, b in zip(raw, mad):
            w.writerow([a, b])
    print(f'  Saved: {save_dir}', flush=True)
    print(f'\n[DONE] OOD={fam}  MAD F1={mad_res["best_f1"]:.4f}  ThreshFree F1={tf_res["best_f1"]:.4f}  '
          f'ROC={tf_res["roc_auc"]:.4f}', flush=True)


# ======================================================================
# AGGREGATE
# ======================================================================
@_tee_to_file('aggregate_summary.txt')
def aggregate(result_dir):
    rows, missing = [], []
    for r in range(3):
        for s in range(N_SPLITS):
            p = os.path.join(result_dir, f'run{r}', f'split{s}', 'summary.json')
            if not os.path.exists(p):
                missing.append((r, s)); continue
            d = json.load(open(p))
            rows.append((s, d['ood_family'], d['MAD_Threshold']['best_f1'], d['Threshold_Free']['best_f1'],
                         d['Threshold_Free']['roc_auc'], d['Threshold_Free']['pr_auc'], d['roc_auc_mad_score'],
                         d.get('baselines_same_input', {}).get('pca5_l2norm_centroid_roc', np.nan),
                         d.get('baselines_same_input', {}).get('input_angles_centroid_roc', np.nan),
                         d['actual_epochs'], d['best_epoch']))
    print('=' * 88)
    print('Group18 Re-uploading Quantum Metric Learning (4-class) -- AGGREGATED RESULTS')
    print('=' * 88)
    print(f'Completed: {len(rows)}/{3 * N_SPLITS}   Missing: {missing if missing else "none"}')
    cols = ['MAD_F1', 'TF_F1', 'ROC_raw', 'PR_raw', 'ROC_mad', 'PCA5_ROC', 'Angle_ROC', 'epochs', 'best_ep']
    print(f"  {'family':<14}{'n':>3}" + ''.join(f'{c:>16}' for c in cols))
    fams = sorted({(x[0], x[1]) for x in rows})
    for s, fam in fams:
        A = np.array([x[2:] for x in rows if x[0] == s], float)
        print(f'  {fam:<14}{len(A):>3}' + ''.join(f'{m:>9.3f}+-{sd:<5.3f}' for m, sd in zip(A.mean(0), A.std(0))))
    if rows:
        A = np.array([x[2:] for x in rows], float)
        print(f"  {'ALL':<14}{len(A):>3}" + ''.join(f'{m:>9.3f}+-{sd:<5.3f}' for m, sd in zip(A.mean(0), A.std(0))))
    print('  PCA5_ROC / Angle_ROC = zero-parameter baselines on the same PCA-5 input / the exact input angles.')
    print('  References (4-class LAMDA): PCA32-only ROC 0.796 | Group17 ROC 0.760 | Group15 MAD-score ROC 0.773')
    print('  Untrained ablation: rerun with --max-epochs 0 --results-dir ablation_untrained/results')


def main():
    p = argparse.ArgumentParser(description='Group18: re-uploading quantum metric learning (4-class)')
    p.add_argument('task_id', help="0-11, or 'aggregate'")
    p.add_argument('--data-dir', default=DEFAULT_DATA_DIR)
    p.add_argument('--prefix', default='drebin_new')
    p.add_argument('--results-dir', default=DEFAULT_RESULTS_DIR)
    p.add_argument('--max-epochs', type=int, default=CAE_EPOCHS, help='use e.g. 30 for a pilot')
    p.add_argument('--no-broadcast', action='store_true', help='simulate samples one by one (fallback)')
    a = p.parse_args()
    if a.task_id == 'aggregate':
        aggregate(os.path.expanduser(a.results_dir))
    else:
        tid = int(a.task_id)
        if not 0 <= tid < 3 * N_SPLITS:
            raise ValueError(f'task_id must be 0-{3 * N_SPLITS - 1}')
        run_task(tid, os.path.expanduser(a.data_dir), a.prefix, os.path.expanduser(a.results_dir),
                 a.max_epochs, not a.no_broadcast)


if __name__ == '__main__':
    main()
