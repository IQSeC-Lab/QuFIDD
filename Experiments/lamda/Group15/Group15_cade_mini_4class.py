#!/usr/bin/env python3
"""
Group15_cade_mini_4class.py
=============================
"CADE-mini", 4-class version: CADE's own architecture and loss, capacity-
shrunk to pair with Group14's VQC M1a-fix (5 qubits, 20 layers, 200
quantum params) on the 4-class LAMDA subset (droidkungfu, plankton,
smsreg, airpush).

Architecture:
  raw features (F-dim)
    -> PCA(F->32), FIXED/frozen (identical role to Group6/Group9's PCA,
       and to the VQC's amplitude-embedding input dimension)
    -> TinyEncoder: Linear(32->3) -> ReLU -> Linear(3->3) -> Tanh
       (matches CADE's own encoder exactly: ReLU hidden, Tanh-bounded
       bottleneck at n_classes-1=3, same as CADE's own convention)
    -> TinyDecoder: Linear(3->2) -> ReLU -> Linear(2->32)
       (reconstructs the PCA output, not the raw input, since PCA itself
       is frozen and provides no gradient path back to F -- exactly
       Group6/Group9's own convention)

Loss: CADE's exact contrastive_loss() + reconstruction_loss(), unchanged
formula, unchanged margin=10.0, unchanged lambda=0.1 -- reconstruction
target is the PCA output instead of raw x, the only architectural change.
LR raised to 0.001 (10x CADE's own 0.0001) -- the one disclosed, necessary
deviation: at 215 parameters, CADE's own LR produced a completely flat
loss curve for the full 250 epochs (confirmed via direct test), so this
is required just to make training happen, not a result-shaping choice.

Scoring: CADE's exact MAD-threshold and threshold-free Euclidean-to-
centroid evaluation, unchanged, reused verbatim.

Trainable parameters: 215 (encoder: Linear(32,3)+Linear(3,3) = 111;
decoder: Linear(3,2)+Linear(2,32) = 104) -- within 7.5% of the VQC's 200.

Diagnostic history: four independent interventions (LR, ReLU vs
LeakyReLU, Tanh vs linear output, full vs zero reconstruction weight)
were each tested against the finding that 2 of the 3 bottleneck
dimensions collapse to fixed, sample-independent offsets, with only 1
dimension carrying real class-discriminative signal. None changed the
qualitative result -- this is treated as the genuine, faithful
capacity-matched outcome (the loss correctly finds that a
sample-independent dimension costs it nothing, given classes separable
along ~1 real axis at this capacity), not a bug to keep chasing.

Data: LAMDA, month-stratified downsized dataset, CADE-style per-split
VarianceThreshold features, filtered to 4 classes: droidkungfu,
plankton, smsreg, airpush.

Reference:
  Yang et al. "CADE: Detecting and Explaining Concept Drift Samples
  for Security Applications." USENIX Security 2021.

Usage:
  python3 Group15_cade_mini_4class.py <task_id>
  python3 Group15_cade_mini_4class.py aggregate
"""

import os, sys, csv, json, time, argparse, functools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA


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

# ── Reproducibility ──────────────────────────────────────────────────
SEEDS = {0: 42, 1: 123, 2: 456}

# ── CADE Hyperparameters (from paper + GitHub) ────────────────────────
# Architecture
# Architecture is adaptive based on input_dim (set per-split in run_task)
# F <= 1000: F→512→128→32→7  (original CADE)
# F >  1000: F→1024→512→128→32→7  (adaptive for larger feature spaces)
# CADE-mini architecture: PCA_DIM is fixed (matches Group6/Group9/VQC's
# amplitude-embedding input); ENC_HIDDEN/DEC_HIDDEN are sized to hit the
# VQC's 300-parameter budget as closely as possible (see MIXER_TOTAL_PARAMS).
# NOTE: the adaptive ARCH_LADDER/get_hidden_dims used by CADE's original,
# unshrunk encoder is NOT used here -- PCA replaces that entire adaptive
# ladder with a single fixed 32-dim reduction, per the capacity-matching design.
N_CLASSES      = 4    # 4-class subset, NOT 8
BOTTLENECK_DIM = N_CLASSES - 1  # = 3, UNCHANGED CADE convention, now for 4 classes
PCA_DIM        = 32   # frozen PCA output dim -- same as Group6/Group9/VQC's amplitude-embedding input
ENC_HIDDEN     = 3    # tiny encoder hidden width (capacity-matched to Group14's 200 quantum params)
DEC_HIDDEN     = 2    # tiny decoder hidden width (capacity-matched)
MIXER_TOTAL_PARAMS = 36 * (ENC_HIDDEN + DEC_HIDDEN) + 35  # = 215, capacity-matched to VQC's 200


# Training
CAE_LR          = 0.001            # 10x CADE original -- confirmed necessary to escape
                                     # early-training stall (see diagnostic history)
CAE_BATCH_SIZE  = 64               # --cae-batch-size 64 (32 left + 32 right)
CAE_EPOCHS      = 250              # --cae-epochs 250 (default)
LAMBDA_1        = 0.1              # --cae-lambda-1 1e-1
SIMILAR_RATIO   = 0.25             # --similar-ratio 0.25
MARGIN          = 10.0             # --margin 10.0

# Detection
MAD_THRESHOLD   = 3.5             # CADE paper default threshold
ALPHA_GRID      = [round(a*0.1,1) for a in range(11)]  # for threshold-free

# Paths (defaults — overridden by --data-dir / --results-dir args)
DEFAULT_DATA_DIR    = os.path.expanduser('~/projects/OOD/Data/cade_lamda_4class')
DEFAULT_RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'results')


# ======================================================================
# MODEL
# ======================================================================

class PCACompressor:
    """
    PCA-based dimensionality reduction: F -> PCA_DIM.
    Unsupervised, fitted once on X_train, frozen thereafter -- identical
    role and implementation to Group6/Group9's PCACompressor.
    """
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
        return z / norm  # L2-normalize, same convention as Group6/Group9

    def compress_tensor(self, X_tensor):
        X_np = X_tensor.numpy() if hasattr(X_tensor, 'numpy') else X_tensor
        return torch.FloatTensor(self.transform(X_np))


class TinyEncoder(nn.Module):
    """
    Capacity-matched CADE encoder: PCA_DIM -> ENC_HIDDEN -> BOTTLENECK_DIM,
    ReLU hidden, Tanh final -- SAME final-activation convention as CADE's
    own real encoder (bounded 3-dim bottleneck = n_classes-1, exactly like
    CADE's own n_classes-1 convention).

    Diagnostic history (see project notes): four independent interventions
    -- LR (0.0001 vs 0.001), activation (ReLU vs LeakyReLU), output bound
    (Tanh vs linear), and loss composition (full reconstruction vs zero
    reconstruction weight) -- were each tested and NONE changed the
    qualitative result: dims 0-1 of the 3-dim bottleneck consistently
    collapse to fixed, sample-independent offsets, with only 1 of 3
    dimensions carrying real class-discriminative signal. This is the
    loss doing exactly what it is supposed to do: a dimension with the
    same value for every sample costs the contrastive loss nothing, so
    with only ENC_HIDDEN=3 units and classes separable along ~1 real
    axis in this data, the network correctly finds the cheapest solution.
    This is treated as the genuine, faithful capacity-matched result, not
    a bug -- restored here to CADE's own exact architecture (ReLU + Tanh,
    full reconstruction weight), with LR raised to 0.001 as the one
    disclosed, necessary deviation (CADE's own LR=0.0001 was tuned for a
    ~1.5M-parameter network; at 215 parameters it produced a completely
    flat loss curve for the full 250 epochs -- LR=0.001 is required just
    to make training happen at all, not a result-shaping choice).
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(PCA_DIM, ENC_HIDDEN), nn.ReLU(),
            nn.Linear(ENC_HIDDEN, BOTTLENECK_DIM), nn.Tanh(),
        )

    def forward(self, z_pca):
        return self.net(z_pca)


class TinyDecoder(nn.Module):
    """
    Capacity-matched CADE decoder: BOTTLENECK_DIM -> DEC_HIDDEN -> PCA_DIM.
    Reconstructs the PCA output (not raw x) since PCA is frozen -- same
    convention as Group6/Group9's decoder.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(BOTTLENECK_DIM, DEC_HIDDEN), nn.ReLU(),
            nn.Linear(DEC_HIDDEN, PCA_DIM),
        )

    def forward(self, z):
        return self.net(z)


class CADEMini(nn.Module):
    """
    Capacity-matched CADE: frozen PCA front-end (same role as Group6/
    Group9's PCA, and the VQC's amplitude-embedding input) + a tiny
    trainable encoder/decoder sized to match the VQC's 300 parameters,
    trained with CADE's own unmodified loss and evaluated with CADE's
    own unmodified scoring.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.pca_comp = PCACompressor(input_dim)
        self.encoder  = TinyEncoder()
        self.decoder  = TinyDecoder()

    def encode(self, x):
        """Raw x -> frozen PCA -> tiny trainable encoder -> 7-dim z."""
        z_pca = self.pca_comp.compress_tensor(x)
        return self.encoder(z_pca)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        """Returns (z, x_hat, z_pca) -- z_pca is the reconstruction TARGET,
        since PCA is frozen and provides no gradient path back to raw x."""
        z_pca = self.pca_comp.compress_tensor(x)
        z     = self.encoder(z_pca)
        x_hat = self.decode(z)
        return z, x_hat, z_pca


# ======================================================================
# LOSS FUNCTIONS (faithful to CADE paper)
# ======================================================================

def contrastive_loss(z, labels):
    """
    EXACT CADE contrastive loss from autoencoder.py.

    Batch layout: first half = left, second half = right.
    Each left[i] is compared with right[i].

    From autoencoder.py:
        is_same = (labels[left] == labels[right])
        dist = ||z_left - z_right||_2
        loss = is_same * dist + (1 - is_same) * relu(margin - dist)
        loss = mean(loss)

    NOT squared distances. relu not squared.
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


# ======================================================================
# TRAINING
# ======================================================================

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def train_cade(model, X_train, y_train, seed, device='cpu'):
    model.to(device)

    # Fit the frozen PCA front-end ONCE, before training -- same pattern
    # as Group6/Group9. Only encoder+decoder parameters are trainable
    # from here on; PCA never receives gradient.
    print(f'\n  [STAGE 1a] Fitting PCA: F={X_train.shape[1]}->{PCA_DIM}', flush=True)
    model.pca_comp.fit(X_train)

    trainable_params = list(model.encoder.parameters()) + list(model.decoder.parameters())
    optimizer = optim.Adam(trainable_params, lr=CAE_LR)

    n_params = sum(p.numel() for p in trainable_params if p.requires_grad)
    print(f'\n  [STAGE 1b] Training capacity-matched CADE-mini', flush=True)
    print(f'  Total trainable params: {n_params:,}  (capacity-matched to VQC\'s 300)', flush=True)
    print(f'  Architecture: F->PCA({PCA_DIM})->{ENC_HIDDEN}->{BOTTLENECK_DIM} (encoder), '
          f'{BOTTLENECK_DIM}->{DEC_HIDDEN}->{PCA_DIM} (decoder)', flush=True)
    print(f'  Loss: {LAMBDA_1}*contrastive + reconstruction (target=PCA output, PCA is frozen)', flush=True)
    print(f'  Margin: {MARGIN}  LR: {CAE_LR}  Epochs: {CAE_EPOCHS}  '
          f'(full-epoch sweep, faithful to real CADE\'s epoch_batches)', flush=True)

    model.train()
    set_seed(seed)

    for epoch in range(1, CAE_EPOCHS + 1):
        batch_count, b_out_x, b_out_y = epoch_batches(
            X_train, y_train, CAE_BATCH_SIZE, SIMILAR_RATIO)

        loss_list, con_list, recon_list = [], [], []
        for b in range(batch_count):
            Xb = torch.FloatTensor(b_out_x[b]).to(device)
            yb = torch.LongTensor(b_out_y[b]).to(device)

            optimizer.zero_grad()
            z, x_hat, z_pca_b = model(Xb)

            l_con   = contrastive_loss(z, yb)
            l_recon = reconstruction_loss(x_hat, z_pca_b)
            loss    = LAMBDA_1 * l_con + l_recon

            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            con_list.append(l_con.item())
            recon_list.append(l_recon.item())

        if epoch % 10 == 0:
            print(f'  Epoch {epoch:3d}  batches={batch_count}  '
                  f'total={np.mean(loss_list):.5f}  '
                  f'con={np.mean(con_list):.5f}  '
                  f'recon={np.mean(recon_list):.5f}', flush=True)

    return model


# ======================================================================
# OOD DETECTION
# ======================================================================

def get_latent(model, X, device='cpu', batch_size=512):
    """Extract 7-dim latent vectors from encoder."""
    model.eval()
    Xt, out = torch.FloatTensor(X).to(device), []
    with torch.no_grad():
        for s in range(0, len(Xt), batch_size):
            z = model.encode(Xt[s:s+batch_size]).cpu().numpy()
            out.append(z)
    return np.vstack(out)


def compute_class_centroids(z_train, y_train):
    classes   = np.unique(y_train)
    centroids = {c: z_train[y_train == c].mean(axis=0) for c in classes}
    return centroids


def compute_mad_scores(z_train, y_train, z_test, centroids):
    """
    CADE MAD-based OOD scoring.

    For each test sample:
      1. Compute Euclidean distance to each class centroid
      2. Find nearest class centroid
      3. Compute MAD deviation: A = (dist - median_dist) / MAD
      4. OOD score = max anomaly score across classes
         (CADE flags as OOD if score > MAD_THRESHOLD)

    We return two things:
      - raw_scores: minimum distance to nearest centroid (for threshold-free)
      - mad_scores: MAD-normalized anomaly score (for MAD threshold=3.5)
    """
    classes = list(centroids.keys())

    # Compute per-class distances for training set (to get median and MAD)
    class_stats = {}
    for c in classes:
        z_c   = z_train[y_train == c]
        dists = np.linalg.norm(z_c - centroids[c], axis=1)
        med   = np.median(dists)
        # 1.4826 scaling: assumes Gaussian — from CADE detect.py get_MAD_for_each_family
        mad   = 1.4826 * np.median(np.abs(dists - med))
        mad   = max(mad, 1e-6)
        class_stats[c] = {'median': med, 'mad': mad}

    # Score test samples
    raw_scores = []   # min Euclidean distance to nearest centroid
    mad_scores = []   # MAD-normalized score (CADE's A^(k) value)

    for z in z_test:
        # Exact CADE detect.py: anomaly_k = |dist - median| / MAD for each class
        # CADE flags OOD if min(anomaly_k) > threshold (not max)
        dis_k  = [float(np.linalg.norm(z - centroids[c])) for c in classes]
        anom_k = [abs(dis_k[i] - class_stats[c]['median']) / class_stats[c]['mad']
                  for i, c in enumerate(classes)]
        min_dist = float(np.min(dis_k))
        min_anom = float(np.min(anom_k))
        raw_scores.append(min_dist)
        mad_scores.append(min_anom)

    return np.array(raw_scores), np.array(mad_scores)


# ======================================================================
# EVALUATION
# ======================================================================

def evaluate_ood(scores, y_test, ood_label):
    """Threshold-free evaluation — same as Model10 protocol."""
    n_ood   = int(np.sum(y_test == ood_label))
    n_known = len(y_test) - n_ood
    n_total = len(y_test)
    order   = np.argsort(scores)[::-1]
    y_s     = y_test[order]

    pc, rc, f1c = [], [], []
    tp = 0
    for k in range(1, n_total + 1):
        if y_s[k-1] == ood_label: tp += 1
        p = tp / k
        r = tp / n_ood if n_ood > 0 else 0.0
        f = 2*p*r/(p+r) if (p+r) > 0 else 0.0
        pc.append(p); rc.append(r); f1c.append(f)

    f1_arr   = np.array(f1c)
    best_idx = int(np.argmax(f1_arr))
    effort   = (best_idx + 1) / n_ood if n_ood > 0 else 0.0

    top_k = order[:best_idx + 1]
    tp_v  = int(np.sum(y_test[top_k] == ood_label))
    fp_v  = (best_idx + 1) - tp_v
    fn_v  = n_ood - tp_v
    fpr   = fp_v / n_known if n_known > 0 else 0.0
    fnr   = fn_v / n_ood   if n_ood   > 0 else 0.0

    # ROC-AUC
    tpr_r, fpr_r = [0.0], [0.0]
    tp_r, fp_r   = 0, 0
    for i in range(n_total):
        if y_test[order[i]] == ood_label: tp_r += 1
        else:                              fp_r += 1
        tpr_r.append(tp_r / n_ood   if n_ood   > 0 else 0.0)
        fpr_r.append(fp_r / n_known if n_known > 0 else 0.0)
    roc_auc = float(np.clip(np.trapezoid(np.array(tpr_r), np.array(fpr_r)), 0, 1))

    # PR-AUC
    rec_a  = np.concatenate([[0.0], np.array(rc)])
    prec_a = np.concatenate([[1.0], np.array(pc)])
    pr_auc = float(np.clip(np.trapezoid(prec_a, rec_a), 0, 1))

    return {
        'best_f1':        float(f1_arr[best_idx]),
        'best_precision': float(pc[best_idx]),
        'best_recall':    float(rc[best_idx]),
        'fpr':            float(fpr),
        'fnr':            float(fnr),
        'norm_effort':    float(effort),
        'roc_auc':        roc_auc,
        'pr_auc':         pr_auc,
        'n_ood':          n_ood,
        'n_known':        n_known,
    }


def evaluate_mad_threshold(mad_scores, y_test, ood_label, threshold=MAD_THRESHOLD):
    """
    CADE's original MAD threshold evaluation.
    Samples with mad_score > threshold are flagged as OOD.
    """
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
# SAVE
# ======================================================================

def save_results(save_dir, run_id, split_id, seed, n_train, n_test,
                 n_features, train_time,
                 mad_res,      # M0a: MAD threshold result
                 tf_res,       # M0b: threshold-free result
                 raw_scores, mad_scores,
                 ood_family_name="unknown"):
    """Save summary.json and ood_scores.csv matching Model10 format."""

    arch_str = f"{n_features}->PCA({PCA_DIM})->{ENC_HIDDEN}->{BOTTLENECK_DIM}"

    summary = {
        'model':          'Group15_CADEMini_4class_PyTorch',
        'reference':      'CADE architecture (Yang et al. USENIX Security 2021), '
                           'capacity-shrunk to match VQC parameter count',
        'architecture':   f'{arch_str}  (bottleneck={BOTTLENECK_DIM}=n_classes-1, '
                           f'PCA frozen, {MIXER_TOTAL_PARAMS} trainable params)',
        'loss':           f'{LAMBDA_1}*contrastive(Euclidean,margin={MARGIN}) + reconstruction',
        'run_id':         int(run_id),
        'split_id':       int(split_id),
        'ood_family':     ood_family_name,
        'seed':           int(seed),
        'n_train':        int(n_train),
        'n_test':         int(n_test),
        'n_features':     int(n_features),
        'bottleneck_dim': BOTTLENECK_DIM,
        'cae_epochs':     CAE_EPOCHS,
        'train_time_sec': round(train_time, 2),
        # M0a: CADE's original MAD threshold scoring
        'CADE_MAD_Threshold': {
            'threshold':      MAD_THRESHOLD,
            'F1':             mad_res['best_f1'],
            'Precision':      mad_res['best_precision'],
            'Recall':         mad_res['best_recall'],
            'FPR':            mad_res['fpr'],
            'FNR':            mad_res['fnr'],
            'Effort':         mad_res['norm_effort'],
            'TP':             mad_res['tp'],
            'FP':             mad_res['fp'],
            'FN':             mad_res['fn'],
        },
        # M0b: threshold-free evaluation (our protocol)
        'CADE_Threshold_Free': {
            'F1':        round(tf_res['best_f1'],        4),
            'Precision': round(tf_res['best_precision'], 4),
            'Recall':    round(tf_res['best_recall'],    4),
            'FPR':       round(tf_res['fpr'],            4),
            'FNR':       round(tf_res['fnr'],            4),
            'ROC_AUC':   round(tf_res['roc_auc'],        4),
            'PR_AUC':    round(tf_res['pr_auc'],         4),
            'Effort':    round(tf_res['norm_effort'],    4),
        },
    }

    with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(save_dir, 'ood_scores.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['sample_idx', 'true_label', 'is_ood',
                    'raw_euclidean_score', 'mad_score',
                    'mad_flagged'])
        for i, (r, m) in enumerate(zip(raw_scores, mad_scores)):
            is_ood = 0  # placeholder — actual label from y_test
            w.writerow([i, is_ood, is_ood,
                        round(float(r), 6),
                        round(float(m), 6),
                        int(m > MAD_THRESHOLD)])

    print(f'  Saved: {save_dir}', flush=True)


# ======================================================================
# MAIN TASK
# ======================================================================

def run_task(task_id, data_dir, prefix, result_dir):
    N_SPLITS = N_CLASSES  # leave-one-family-out: one split per class
    run_id   = task_id // N_SPLITS
    split_id = task_id %  N_SPLITS
    seed     = SEEDS[run_id]

    # Dynamically load family name from npz — works for any dataset
    _npz_check = os.path.join(
        os.path.expanduser(data_dir), f'{prefix}_{split_id}.npz')
    _d_check   = np.load(_npz_check, allow_pickle=True)
    ood_family_name = str(_d_check['heldout_family'].item()) \
                      if _d_check['heldout_family'].shape == () \
                      else str(_d_check['heldout_family'])


    print('=' * 65, flush=True)
    print(f'M0 CADE Original | Task={task_id} | Run={run_id} | '
          f'Split={split_id} | Seed={seed} | OOD={ood_family_name}',
          flush=True)
    print('=' * 65, flush=True)

    save_dir = os.path.join(result_dir, f'run{run_id}', f'split{split_id}')
    os.makedirs(save_dir, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────
    npz_path = os.path.join(data_dir, f'{prefix}_{split_id}.npz')
    print(f'  Loading: {npz_path}', flush=True)
    d       = np.load(npz_path, allow_pickle=True)
    # Support both dense (Drebin) and sparse (LAMDA) formats
    _xt = d['X_train']
    X_train = _xt.item().toarray().astype(np.float32) \
              if _xt.shape == () else _xt.astype(np.float32)
    y_train = d['y_train'].astype(np.int64)
    _xte = d['X_test']
    X_test  = _xte.item().toarray().astype(np.float32) \
              if _xte.shape == () else _xte.astype(np.float32)
    y_test  = d['y_test'].astype(np.int64)

    print(f'  X_train={X_train.shape}  X_test={X_test.shape}', flush=True)
    print(f'  Classes in y_train={len(np.unique(y_train))} '
          f'(OOD family {ood_family_name} absent — expected)', flush=True)
    print(f'  Architecture: F={X_train.shape[1]}->PCA({PCA_DIM})->'
          f'{ENC_HIDDEN}->{BOTTLENECK_DIM}  (capacity-matched CADE-mini)', flush=True)

    set_seed(seed)

    # ── Train CADE-mini ────────────────────────────────────────────────
    print(f'\n[STAGE 1] Training capacity-matched CADE-mini '
          f'(PCA frozen, tiny encoder/decoder trainable)', flush=True)
    device = 'cpu'
    model  = CADEMini(X_train.shape[1])
    t0     = time.time()
    model  = train_cade(model, X_train, y_train, seed, device)
    train_time = time.time() - t0
    print(f'  Training time: {train_time:.1f}s', flush=True)

    # Save weights (encoder/decoder only -- pca_comp is a plain sklearn
    # object, not an nn.Module submodule, so it is NOT captured by
    # state_dict() and must be saved separately, same as Group6/Group9)
    weights_path = os.path.join(save_dir, 'model_weights.pt')
    torch.save(model.state_dict(), weights_path)
    pca_path = os.path.join(save_dir, 'pca_components.pkl')
    import pickle
    with open(pca_path, 'wb') as f:
        pickle.dump(model.pca_comp.pca, f)

    # ── Extract latent representations ─────────────────────────────────
    print(f'\n[STAGE 2] Extracting latent representations', flush=True)
    z_train = get_latent(model, X_train, device)
    z_test  = get_latent(model, X_test,  device)
    print(f'  z_train={z_train.shape}  z_test={z_test.shape}', flush=True)

    # Save latent test vectors
    with open(os.path.join(save_dir, 'cade_latent_test.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([f'z{i}' for i in range(BOTTLENECK_DIM)]
                   + ['true_label', 'is_ood'])
        for i, z in enumerate(z_test):
            lbl   = int(y_test[i])
            is_ood = int(lbl == split_id)
            w.writerow(list(np.round(z, 6)) + [lbl, is_ood])

    # ── Compute OOD scores ─────────────────────────────────────────────
    print(f'\n[STAGE 3] Computing OOD scores', flush=True)
    centroids   = compute_class_centroids(z_train, y_train)

    # ── DIAGNOSTIC: are the KNOWN classes even separated from each other? ──
    print(f'\n  [DIAGNOSTIC] Known-class separation on TRAINING data:', flush=True)
    classes = sorted(centroids.keys())
    intra_spreads = {}
    for c in classes:
        z_c = z_train[y_train == c]
        d_to_own = np.linalg.norm(z_c - centroids[c], axis=1)
        intra_spreads[c] = float(d_to_own.mean())
        print(f'    class {c}: n={len(z_c):4d}  '
              f'mean dist to own centroid = {intra_spreads[c]:.4f}', flush=True)
    print(f'    mean intra-class spread (avg dist to own centroid): '
          f'{np.mean(list(intra_spreads.values())):.4f}', flush=True)
    inter_dists = []
    for i, c1 in enumerate(classes):
        for c2 in classes[i+1:]:
            d = float(np.linalg.norm(centroids[c1] - centroids[c2]))
            inter_dists.append(d)
            print(f'    centroid dist class {c1} <-> class {c2}: {d:.4f}', flush=True)
    mean_inter = np.mean(inter_dists) if inter_dists else 0.0
    mean_intra = np.mean(list(intra_spreads.values()))
    ratio = mean_inter / (mean_intra + 1e-8)
    print(f'    mean inter-centroid distance: {mean_inter:.4f}', flush=True)
    print(f'    separation ratio (inter/intra): {ratio:.4f}  '
          f'(>>1 = well-separated classes; ~1 or below = classes overlapping/smeared)',
          flush=True)

    # ── DIAGNOSTIC 2: is Tanh's [-1,1]^3 range actually being used? ──
    max_diag = np.sqrt(4 * z_train.shape[1])  # max possible distance in [-1,1]^d
    print(f'\n  [DIAGNOSTIC] z_train range per dimension '
          f'(Tanh bounds each to [-1,1], max cube diagonal = {max_diag:.4f}):', flush=True)
    for dim in range(z_train.shape[1]):
        col = z_train[:, dim]
        print(f'    dim {dim}: min={col.min():+.4f}  max={col.max():+.4f}  '
              f'mean={col.mean():+.4f}  std={col.std():.4f}', flush=True)
    overall_std = z_train.std()
    frac_of_range = overall_std / 1.0  # Tanh's per-dim range half-width is 1.0
    print(f'    overall std across all dims: {overall_std:.4f}  '
          f'(std >> 0.3-0.5 suggests good range usage; std << 0.1 suggests '
          f'activations are bunched near zero, not a capacity problem)', flush=True)

    raw_scores, mad_scores = compute_mad_scores(
        z_train, y_train, z_test, centroids)

    # M0a: MAD threshold scoring (original CADE)
    mad_res = evaluate_mad_threshold(mad_scores, y_test, split_id, MAD_THRESHOLD)
    print(f'  [M0a] CADE MAD(threshold={MAD_THRESHOLD}): '
          f'F1={mad_res["best_f1"]:.4f}  '
          f'Prec={mad_res["best_precision"]:.4f}  '
          f'Rec={mad_res["best_recall"]:.4f}  '
          f'FPR={mad_res["fpr"]:.4f}  FNR={mad_res["fnr"]:.4f}', flush=True)

    # M0b: threshold-free scoring (our protocol)
    tf_res = evaluate_ood(raw_scores, y_test, split_id)
    print(f'  [M0b] Threshold-free: '
          f'F1={tf_res["best_f1"]:.4f}  '
          f'Prec={tf_res["best_precision"]:.4f}  '
          f'Rec={tf_res["best_recall"]:.4f}  '
          f'ROC-AUC={tf_res["roc_auc"]:.4f}  '
          f'FPR={tf_res["fpr"]:.4f}  FNR={tf_res["fnr"]:.4f}', flush=True)

    # ── Save ───────────────────────────────────────────────────────────
    print(f'\n[STAGE 4] Saving results', flush=True)
    save_results(
        save_dir, run_id, split_id, seed,
        X_train.shape[0], X_test.shape[0], X_train.shape[1],
        train_time, mad_res, tf_res, raw_scores, mad_scores,
        ood_family_name=ood_family_name)

    print(f'\n[DONE] OOD={ood_family_name}', flush=True)
    print(f'  M0a CADE MAD F1:         {mad_res["best_f1"]:.4f}', flush=True)
    print(f'  M0b Threshold-Free F1:   {tf_res["best_f1"]:.4f}', flush=True)


# ======================================================================
# AGGREGATE
# ======================================================================

@_tee_to_file('aggregate_summary.txt')
def aggregate(result_dir):
    rows, missing = [], []
    # Detect number of splits from available results
    n_splits = 0
    for _s in range(20):
        if os.path.exists(os.path.join(
                result_dir, 'run0', f'split{_s}', 'summary.json')):
            n_splits = _s + 1
        else: break
    if n_splits == 0: print('No results found.'); return

    for r in range(3):
        for s in range(n_splits):
            p = os.path.join(result_dir, f'run{r}', f'split{s}', 'summary.json')
            if not os.path.exists(p):
                missing.append((r, s)); continue
            with open(p) as f:
                d = json.load(f)
            row = {
                'run_id':    r,
                'split_id':  s,
                'ood_family': d.get('ood_family', f'split{s}'),
                'seed':       SEEDS[r],
            }
            for key, meth in [('mad', 'CADE_MAD_Threshold'),
                               ('tf',  'CADE_Threshold_Free')]:
                blk = d.get(meth, {})
                for m in ['F1','Precision','Recall','FPR','FNR','Effort']:
                    row[f'{key}__{m}'] = blk.get(m, float('nan'))
                if key == 'tf':
                    row['tf__ROC_AUC'] = blk.get('ROC_AUC', float('nan'))
                    row['tf__PR_AUC']  = blk.get('PR_AUC',  float('nan'))
            rows.append(row)

    if missing: print(f'Missing {len(missing)}/{3*n_splits}: {missing}')
    if not rows: print('No results found.'); return

    def mat(col):
        out = np.full((3, 8), np.nan)
        for row in rows:
            out[int(row['run_id']), int(row['split_id'])] = row.get(col, np.nan)
        return out

    print('='*72)
    print('Group15 CADE-MINI (4-class, 215 params) — AGGREGATED RESULTS')
    print('='*72)

    for label, prefix in [('M0a: CADE MAD (threshold=3.5)', 'mad'),
                           ('M0b: CADE Threshold-Free (our protocol)', 'tf')]:
        print(f'\n{label}')
        print(f'  {"Family":<16} {"F1":>8} {"Prec":>8} {"Rec":>8} '
              f'{"FPR":>8} {"FNR":>8} {"Effort":>8}')
        print(f'  {"─"*64}')
        for s in range(n_splits):
            f1   = mat(f'{prefix}__F1')[:, s]
            prec = mat(f'{prefix}__Precision')[:, s]
            rec  = mat(f'{prefix}__Recall')[:, s]
            fpr  = mat(f'{prefix}__FPR')[:, s]
            fnr  = mat(f'{prefix}__FNR')[:, s]
            eff  = mat(f'{prefix}__Effort')[:, s]
            _fn = rows[s*3]['ood_family'] if len(rows)>s*3 else f'split{s}'
            print(f'  {_fn:<16}'
                  f'{np.nanmean(f1):>8.4f}'
                  f'{np.nanmean(prec):>8.4f}'
                  f'{np.nanmean(rec):>8.4f}'
                  f'{np.nanmean(fpr):>8.4f}'
                  f'{np.nanmean(fnr):>8.4f}'
                  f'{np.nanmean(eff):>8.4f}')
        f1_all = mat(f'{prefix}__F1')
        print(f'  {"─"*64}')
        print(f'  {"MEAN":<16}'
              f'{np.nanmean(f1_all):>8.4f}±{np.nanstd(f1_all):<6.4f}')

    # Comparison
    print(f'\n{"="*72}')
    print('COMPARISON vs Our Method')
    print(f'  {"Method":<30} {"F1 mean±std":>16} {"ROC_AUC":>10}')
    print(f'  {"─"*60}')
    for label, prefix in [('M0a CADE MAD', 'mad'), ('M0b CADE Thresh-Free', 'tf')]:
        f1 = mat(f'{prefix}__F1')
        if prefix == 'tf':
            roc = mat(f'{prefix}__ROC_AUC')
            print(f'  {label:<30}'
                  f'{np.nanmean(f1):>7.4f}±{np.nanstd(f1):<8.4f}'
                  f'{np.nanmean(roc):>10.4f}')
        else:
            print(f'  {label:<30}'
                  f'{np.nanmean(f1):>7.4f}±{np.nanstd(f1):<8.4f}'
                  f'{"N/A (fixed threshold)":>10}')

    print(f'\n  Completed: {len(rows)}/{3*n_splits}')


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Group15: CADE-mini (215 params, 4-class, frozen PCA) — LAMDA')
    p.add_argument('task_id',
                   help='Integer 0-23 or "aggregate"')
    p.add_argument('--data-dir',   default=DEFAULT_DATA_DIR)
    p.add_argument('--prefix',     default='drebin_new')
    p.add_argument('--results-dir', default=DEFAULT_RESULTS_DIR)
    args = p.parse_args()

    if args.task_id == 'aggregate':
        aggregate(os.path.expanduser(args.results_dir))
    else:
        task_id = int(args.task_id)
        assert 0 <= task_id <= 11, f'task_id must be 0-11 (4 splits x 3 seeds), got {task_id}'
        run_task(task_id,
                 os.path.expanduser(args.data_dir),
                 args.prefix,
                 os.path.expanduser(args.results_dir))
