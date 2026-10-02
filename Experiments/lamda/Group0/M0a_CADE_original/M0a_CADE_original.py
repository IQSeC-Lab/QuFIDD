#!/usr/bin/env python3
"""
M0a_CADE_LAMDA_monthwise.py
============================
Faithful PyTorch reimplementation of CADE (USENIX Security 2021).
Original uses TensorFlow/Keras — this reimplements in PyTorch for
compatibility with OOD_env, keeping identical architecture and hyperparameters.

Data: LAMDA, month-stratified downsized dataset (max 1000 samples/family,
preserving each family's original year-month distribution), built by
build_cade_splits_monthwise.py. Families: droidkungfu, plankton, basebridge,
smsreg, airpush, fakeinst, shedun, smssend (Drebin-first-then-APIGraph
selection — see README.md in lamda_ood_like_monthwise/).

Architecture (from CADE paper + SEED paper confirmation):
  Encoder: F -> 512 -> 128 -> 32 -> 7  (ReLU after each except last, Tanh at end)
  Decoder: 7 -> 32 -> 128 -> 512 -> F  (mirror)
  Loss:    lambda_1 * contrastive_loss + reconstruction_loss
  Score:   MAD-based Euclidean distance to nearest class centroid

M0a: CADE with original MAD threshold=3.5 (as reported in paper)
M0b: CADE with threshold-free evaluation (our protocol, best F1 across thresholds)
     — same script, different scoring section

Reference:
  Yang et al. "CADE: Detecting and Explaining Concept Drift Samples
  for Security Applications." USENIX Security 2021.

Usage:
  python3 M0a_CADE_original.py <task_id>
  python3 M0a_CADE_original.py aggregate
"""

import os, sys, csv, json, time, argparse, functools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


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
# Ladder-based adaptive architecture:
# Start from nearest ladder dim below F, compress down to 32, then 7.
# e.g. F=2642 → 2200→1800→1400→1024→512→128→32→7
# e.g. F=1655 → 1400→1024→512→128→32→7
# e.g. F=1340 → 1024→512→128→32→7  (Drebin)
ARCH_LADDER    = [2200, 1800, 1400, 1024, 512, 128, 32]
BOTTLENECK_DIM = 7  # n_classes - 1

def get_hidden_dims(input_dim):
    """Ladder rule: nearest dim below input_dim, down to 32."""
    for i, d in enumerate(ARCH_LADDER):
        if d < input_dim:
            return ARCH_LADDER[i:]
    return [32]


# Training
CAE_LR          = 0.0001           # --cae-lr 0.0001 (run_drebin_cade.sh)
CAE_BATCH_SIZE  = 64               # --cae-batch-size 64 (32 left + 32 right)
CAE_EPOCHS      = 250              # --cae-epochs 250 (default)
LAMBDA_1        = 0.1              # --cae-lambda-1 1e-1
SIMILAR_RATIO   = 0.25             # --similar-ratio 0.25
MARGIN          = 10.0             # --margin 10.0

# Detection
MAD_THRESHOLD   = 3.5             # CADE paper default threshold
ALPHA_GRID      = [round(a*0.1,1) for a in range(11)]  # for threshold-free

# Paths (defaults — overridden by --data-dir / --results-dir args)
DEFAULT_DATA_DIR    = os.path.expanduser('~/projects/OOD/Data/cade_lamda')
DEFAULT_RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'results')


# ======================================================================
# MODEL
# ======================================================================

class CADEEncoder(nn.Module):
    """CADE encoder: adaptive hidden dims based on input_dim."""
    def __init__(self, input_dim):
        super().__init__()
        hidden = get_hidden_dims(input_dim)
        dims   = [input_dim] + hidden + [BOTTLENECK_DIM]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Tanh())   # final layer: Tanh
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CADEDecoder(nn.Module):
    """CADE decoder: mirror of encoder, adaptive."""
    def __init__(self, output_dim, input_dim):
        super().__init__()
        hidden = get_hidden_dims(input_dim)
        dims   = [BOTTLENECK_DIM] + list(reversed(hidden)) + [output_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
            # Last layer: no activation (reconstruction to binary space)
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class CADE(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = CADEEncoder(input_dim)
        self.decoder = CADEDecoder(input_dim, input_dim)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z    = self.encode(x)
        x_hat = self.decode(z)
        return z, x_hat


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


def make_cade_batch(X, y, batch_size, similar_ratio):
    """
    CADE batch construction (mirrors data.py epoch_batches).
    Produces batch_size samples: first half=left, second half=right.
    similar_ratio fraction of pairs are same-class.
    """
    half       = batch_size // 2
    n_sim      = int(half * similar_ratio)
    n_diff     = half - n_sim
    classes    = np.unique(y)
    idx_by_cls = {c: np.where(y == c)[0] for c in classes}
    left_idx, right_idx = [], []

    sim_count = 0
    while sim_count < n_sim:
        c = np.random.choice(classes)
        pool = idx_by_cls[c]
        if len(pool) < 2:
            continue
        i, j = np.random.choice(pool, 2, replace=False)
        left_idx.append(i)
        right_idx.append(j)
        sim_count += 1

    diff_count = 0
    while diff_count < n_diff:
        c1, c2 = np.random.choice(classes, 2, replace=False)
        p1, p2 = idx_by_cls[c1], idx_by_cls[c2]
        if len(p1) == 0 or len(p2) == 0:
            continue
        left_idx.append(np.random.choice(p1))
        right_idx.append(np.random.choice(p2))
        diff_count += 1

    idx = np.array(left_idx + right_idx)
    return X[idx], y[idx]


def train_cade(model, X_train, y_train, seed, device='cpu'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=CAE_LR)
    X_t = torch.FloatTensor(X_train).to(device)
    y_t = torch.LongTensor(y_train)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Total params: {n_params:,}', flush=True)
    print(f'  Architecture: F->{"-".join(str(h) for h in get_hidden_dims(X_train.shape[1]))}->{BOTTLENECK_DIM}',
          flush=True)
    print(f'  Loss: {LAMBDA_1}*contrastive + reconstruction', flush=True)
    print(f'  Margin: {MARGIN}  LR: {CAE_LR}  Epochs: {CAE_EPOCHS}', flush=True)

    model.train()
    set_seed(seed)

    for epoch in range(1, CAE_EPOCHS + 1):
        # Build pair batch
        Xb_np, yb_np = make_cade_batch(
            X_train, y_train, CAE_BATCH_SIZE, SIMILAR_RATIO)
        Xb = torch.FloatTensor(Xb_np).to(device)
        yb = torch.LongTensor(yb_np).to(device)

        optimizer.zero_grad()
        z, x_hat = model(Xb)

        l_con   = contrastive_loss(z, yb)
        l_recon = reconstruction_loss(x_hat, Xb)
        loss    = LAMBDA_1 * l_con + l_recon

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'  Epoch {epoch:3d}  '
                  f'total={loss.item():.5f}  '
                  f'con={l_con.item():.5f}  '
                  f'recon={l_recon.item():.5f}', flush=True)

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

    arch_ladder_str = "->".join(
        [str(n_features)] + [str(h) for h in get_hidden_dims(n_features)]
        + [str(BOTTLENECK_DIM)]
    )

    summary = {
        'model':          'M0_CADE_Original_PyTorch',
        'reference':      'Yang et al. USENIX Security 2021',
        'architecture':   f'{arch_ladder_str}  (bottleneck={BOTTLENECK_DIM}=n_classes-1)',
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
    run_id   = task_id // 8
    split_id = task_id %  8
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
    print(f'  Architecture: F={X_train.shape[1]}->'
          f'{"->".join(str(h) for h in get_hidden_dims(X_train.shape[1]))}->{BOTTLENECK_DIM}', flush=True)

    set_seed(seed)

    # ── Train CADE ─────────────────────────────────────────────────────
    print(f'\n[STAGE 1] Training CADE contrastive autoencoder', flush=True)
    device = 'cpu'
    model  = CADE(X_train.shape[1])
    t0     = time.time()
    model  = train_cade(model, X_train, y_train, seed, device)
    train_time = time.time() - t0
    print(f'  Training time: {train_time:.1f}s', flush=True)

    # Save weights
    weights_path = os.path.join(save_dir, 'model_weights.pt')
    torch.save(model.state_dict(), weights_path)

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

    if missing: print(f'Missing {len(missing)}/24: {missing}')
    if not rows: print('No results found.'); return

    def mat(col):
        out = np.full((3, 8), np.nan)
        for row in rows:
            out[int(row['run_id']), int(row['split_id'])] = row.get(col, np.nan)
        return out

    print('='*72)
    print('M0 CADE ORIGINAL — AGGREGATED RESULTS (24 splits)')
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

    print(f'\n  Completed: {len(rows)}/24')


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='M0a CADE Original + M0b Threshold-Free — LAMDA (month-stratified)')
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
        assert 0 <= task_id <= 23, f'task_id must be 0-23, got {task_id}'
        run_task(task_id,
                 os.path.expanduser(args.data_dir),
                 args.prefix,
                 os.path.expanduser(args.results_dir))
