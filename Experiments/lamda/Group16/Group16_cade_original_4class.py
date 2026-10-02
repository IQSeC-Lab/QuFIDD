#!/usr/bin/env python3
"""
Group16_cade_original_4class.py
===============================
CADE exactly as released (Yang et al., USENIX Security 2021), reimplemented in
PyTorch, on the 4-class LAMDA / Drebin leave-one-family-out splits. NO PCA:
the contrastive autoencoder sees the full variance-thresholded feature vector.

Every setting below was checked against CADE's source code
(github.com/whyisyoung/CADE: cade/autoencoder.py, cade/data.py,
cade/detect.py, cade/evaluate.py, cade/utils.py, run_drebin_cade.sh).

ARCHITECTURE  (autoencoder.py: Autoencoder.build)
  dims = [F, 512, 128, 32, N]   F = #features, N = #TRAINING classes
         (main.py: NUM_CLASSES = len(np.unique(y_train)); here N = 3)
  Encoder: Dense+ReLU for hidden layers; the bottleneck layer is LINEAR
           ("no activation is applied here, i.e., linear activation").
  Decoder: mirror; Dense+ReLU hidden layers, LINEAR output layer.
  Init: glorot_uniform weights, zero biases (Keras defaults).
  NOTE: earlier project scripts (Group0/M0a, Group15) used a Tanh bottleneck;
  CADE's released code does not.

LOSS  (autoencoder.py: ContrastiveAE.train)
  dist     = sqrt(sum((z_left - z_right)^2) + 1e-10)       (NOT squared)
  con      = mean( same*dist + (1-same)*relu(margin - dist) )
  ae       = mean over samples of MSE(x, x_hat)
  loss     = lambda_1 * con + ae,  lambda_1 = 0.1, margin = 10.0

TRAINING  (run_drebin_cade.sh + utils.parse_args + data.epoch_batches)
  Adam, lr = 0.0001 (Keras epsilon 1e-7), batch_size = 64 (32 anchors + 32
  partners), similar_ratio = 0.25 (num_sim = int(64*0.25) = 16 same-class
  partners), 250 epochs, every training sample is an anchor once per epoch.
  Checkpointing exactly as CADE: loss is averaged over the batches since the
  last report; every display_interval = 10 epochs (epoch % 10 == 0, epochs
  counted from 0) the average is compared with the best so far and the
  weights are saved if it improved. The model used for detection is that
  best checkpoint. No early stopping.
  (--protocol matched instead uses the rule of Groups 14/17/18: per-epoch
  loss, patience 5, relative min-delta 1e-4, best-epoch weights.)

DETECTION  (detect.py)
  Centroid c_i = mean latent vector of training family i.
  d_i(x) = ||z(x) - c_i||_2 ; per family median and MAD = 1.4826*median|d - median|
  A(x) = min_i |d_i(x) - median_i| / MAD_i ; drifting if A(x) > 3.5.

EVALUATION  (evaluate.py: evaluate_newfamily_as_drift_by_distance)
  CADE protocol (what CADE's Table 3 reports): take the samples with
  A > T_MAD, rank them by min_i d_i (farthest first), walk down the list,
  report the best F1 with its precision, recall and inspection effort.
  Reported at T_MAD = 3.5 (CADE) and T_MAD = 0 (no filter). Also reported:
  fixed-threshold MAD F1, threshold-free metrics on the raw distance and on
  the MAD score, and a zero-parameter baseline (raw-feature centroid ROC).

Output format matches Groups 14/17/18 (summary.json with MAD_Threshold /
Threshold_Free, ood_scores.csv with raw_score, mad_score), so
cade_protocol_eval.py and pca_ensemble_eval.py work on it unchanged.

Usage:
  python3 Group16_cade_original_4class.py <task_id 0-11> --data-dir ... --results-dir results
  python3 Group16_cade_original_4class.py <task_id> --protocol matched --results-dir results_matched
  python3 Group16_cade_original_4class.py aggregate --results-dir results
"""

import os, sys, csv, json, time, copy, argparse, functools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score


# ======================================================================
# CONFIG (CADE's released values)
# ======================================================================
N_CLASSES       = 4
N_SPLITS        = N_CLASSES
SEEDS           = {0: 42, 1: 123, 2: 456}

CAE_HIDDEN      = [512, 128, 32]
CAE_LR          = 0.0001
CAE_BATCH_SIZE  = 64
CAE_EPOCHS      = 250
LAMBDA_1        = 0.1
SIMILAR_RATIO   = 0.25
MARGIN          = 10.0
DISPLAY_INTERVAL = 10
MAD_THRESHOLD   = 3.5
MAD_B           = 1.4826
KERAS_EPS       = 1e-7

# --protocol matched (same stopping rule as Groups 14/17/18)
PATIENCE        = 5
REL_MIN_DELTA   = 1e-4

DEFAULT_DATA_DIR = os.path.expanduser('~/projects/OOD/Data/cade_lamda_4class_vt')


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


# ======================================================================
# MODEL
# ======================================================================
def dense(i, o):
    layer = nn.Linear(i, o)
    nn.init.xavier_uniform_(layer.weight)      # = Keras glorot_uniform
    nn.init.zeros_(layer.bias)                 # = Keras default bias init
    return layer


class CADEAutoencoder(nn.Module):
    def __init__(self, input_dim, n_latent):
        super().__init__()
        dims = [input_dim] + CAE_HIDDEN + [n_latent]
        enc = []
        for i in range(len(dims) - 2):
            enc += [dense(dims[i], dims[i + 1]), nn.ReLU()]
        enc += [dense(dims[-2], dims[-1])]                 # LINEAR bottleneck
        dec = []
        for i in range(len(dims) - 1, 1, -1):
            dec += [dense(dims[i], dims[i - 1]), nn.ReLU()]
        dec += [dense(dims[1], dims[0])]                   # LINEAR output
        self.encoder = nn.Sequential(*enc)
        self.decoder = nn.Sequential(*dec)
        self.dims = dims

    def forward(self, x):
        z = self.encoder(x)
        return z, self.decoder(z)


def contrastive_loss(z, labels):
    half = z.shape[0] // 2
    dist = torch.sqrt(((z[:half] - z[half:]) ** 2).sum(dim=1) + 1e-10)
    same = (labels[:half] == labels[half:]).float()
    return (same * dist + (1.0 - same) * torch.relu(MARGIN - dist)).mean()


def epoch_batches(X_train, y_train, batch_size, similar_ratio):
    """CADE's data.epoch_batches (labels must be 0..N-1, as CADE requires)."""
    if batch_size % 4 != 0:
        raise ValueError('batch_size should be a multiple of 4.')
    half_size   = batch_size // 2
    batch_count = int(X_train.shape[0] / half_size)
    num_sim     = int(batch_size * similar_ratio)
    b_out_x = np.zeros([batch_count, batch_size, X_train.shape[1]], dtype=np.float32)
    b_out_y = np.zeros([batch_count, batch_size], dtype=np.int64)
    random_idx = np.random.permutation(X_train.shape[0])
    b_out_x[:, :half_size] = np.split(X_train[random_idx[: batch_count * half_size]], batch_count)
    b_out_y[:, :half_size] = np.split(y_train[random_idx[: batch_count * half_size]], batch_count)
    index_cls, index_no_cls = [], []
    for label in range(len(np.unique(y_train))):
        index_cls.append(np.where(y_train == label)[0])
        index_no_cls.append(np.where(y_train != label)[0])
    for b in range(batch_count):
        for m in range(0, num_sim):
            pair = np.random.choice(index_cls[b_out_y[b, m]], 1)
            b_out_x[b, m + half_size] = X_train[pair[0]]; b_out_y[b, m + half_size] = y_train[pair[0]]
        for m in range(num_sim, half_size):
            pair = np.random.choice(index_no_cls[b_out_y[b, m]], 1)
            b_out_x[b, m + half_size] = X_train[pair[0]]; b_out_y[b, m + half_size] = y_train[pair[0]]
    return batch_count, b_out_x, b_out_y


# ======================================================================
# TRAINING
# ======================================================================
def train_cade(model, X_tr, y_tr, seed, protocol, max_epochs):
    opt = optim.Adam(model.parameters(), lr=CAE_LR, eps=KERAS_EPS)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Architecture: {"-".join(map(str, model.dims))} (linear bottleneck), '
          f'decoder mirror (linear output). Trainable params: {n_params:,}', flush=True)
    print(f'  Loss: {LAMBDA_1}*contrastive(margin={MARGIN}) + MSE reconstruction', flush=True)
    print(f'  Adam lr={CAE_LR} eps={KERAS_EPS}, batch={CAE_BATCH_SIZE}, similar_ratio={SIMILAR_RATIO}, '
          f'epochs={max_epochs}, protocol={protocol}', flush=True)
    set_seed(seed)
    model.train()
    best_state, best_loss, best_epoch = copy.deepcopy(model.state_dict()), np.inf, -1
    window, since, stopped_early, epochs_run = [], 0, False, max_epochs
    for epoch in range(max_epochs):
        t0 = time.time()
        bc, bx, by = epoch_batches(X_tr, y_tr, CAE_BATCH_SIZE, SIMILAR_RATIO)
        ep_loss, ep_con, ep_ae = [], [], []
        for b in range(bc):
            xb = torch.from_numpy(bx[b]); yb = torch.from_numpy(by[b])
            opt.zero_grad()
            z, x_hat = model(xb)
            l_con = contrastive_loss(z, yb)
            l_ae  = ((x_hat - xb) ** 2).mean(dim=1).mean()
            loss  = LAMBDA_1 * l_con + l_ae
            loss.backward(); opt.step()
            ep_loss.append(loss.item()); ep_con.append(l_con.item()); ep_ae.append(l_ae.item())
        window += ep_loss
        msg = (f'  Epoch {epoch:3d}  loss={np.mean(ep_loss):.5f}  con={np.mean(ep_con):.4f}  '
               f'ae={np.mean(ep_ae):.5f}  ({time.time() - t0:.1f}s)')
        if protocol == 'cade':
            if epoch % DISPLAY_INTERVAL == 0:                      # CADE's checkpoint rule
                cur = float(np.mean(window)); window = []
                if cur < best_loss:
                    best_loss, best_epoch = cur, epoch
                    best_state = copy.deepcopy(model.state_dict())
                    msg += f'  [checkpoint: window loss {cur:.5f}]'
        else:
            cur = float(np.mean(ep_loss))
            if cur < best_loss * (1.0 - REL_MIN_DELTA):
                best_loss, best_epoch, since = cur, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                since += 1
            msg += f'  best={best_loss:.5f} no_improve={since}'
        print(msg, flush=True)
        if protocol == 'matched' and since >= PATIENCE:
            print(f'  [EARLY STOP] epoch {epoch}; best epoch {best_epoch}', flush=True)
            stopped_early, epochs_run = True, epoch + 1
            break
    model.load_state_dict(best_state)
    print(f'  [RESTORE] best checkpoint from epoch {best_epoch} (loss {best_loss:.5f})', flush=True)
    return model, epochs_run, stopped_early, best_epoch, best_loss


# ======================================================================
# DETECTION + EVALUATION
# ======================================================================
def cade_detect(z_tr, y_tr, z_te):
    N = len(np.unique(y_tr))
    cents = [z_tr[y_tr == i].mean(axis=0) for i in range(N)]
    med, mad = [], []
    for i in range(N):
        d = np.linalg.norm(z_tr[y_tr == i] - cents[i], axis=1)
        m = np.median(d); med.append(m)
        mad.append(MAD_B * np.median(np.abs(d - m)))
    mad_safe = [x if x > 0 else 1e-12 for x in mad]            # CADE has no floor; guard only a 0 MAD
    D = np.stack([np.linalg.norm(z_te - c, axis=1) for c in cents], axis=1)
    A = np.stack([np.abs(D[:, i] - med[i]) / mad_safe[i] for i in range(N)], axis=1)
    return D.min(axis=1), A.min(axis=1), {'median': [float(x) for x in med], 'mad': [float(x) for x in mad]}


def cade_protocol(raw, mad, is_ood, T):
    """CADE's evaluate_newfamily_as_drift_by_distance: flagged (A > T) ranked by distance."""
    n_ood, n_known = int(is_ood.sum()), int(len(is_ood) - is_ood.sum())
    idx = np.arange(len(raw)) if T == 0 else np.where(mad > T)[0]
    if len(idx) == 0:
        return {'T_MAD': T, 'n_flagged': 0, 'best_f1': 0.0, 'best_precision': 0.0, 'best_recall': 0.0,
                'fpr': 0.0, 'norm_effort': 0.0}
    order = idx[np.argsort(-raw[idx], kind='stable')]
    tp = np.cumsum(is_ood[order]); k = np.arange(1, len(order) + 1)
    p, r = tp / k, tp / n_ood
    f1 = np.where(p + r > 0, 2 * p * r / (p + r + 1e-12), 0.0); b = int(np.argmax(f1))
    return {'T_MAD': T, 'n_flagged': int(len(idx)), 'best_f1': float(f1[b]), 'best_precision': float(p[b]),
            'best_recall': float(r[b]), 'fpr': float((k[b] - tp[b]) / n_known), 'norm_effort': float(k[b] / n_ood)}


def threshold_free(scores, is_ood):
    n_ood = int(is_ood.sum()); n_known = len(is_ood) - n_ood
    order = np.argsort(-scores, kind='stable'); tp = np.cumsum(is_ood[order]); k = np.arange(1, len(order) + 1)
    p, r = tp / k, tp / n_ood
    f1 = np.where(p + r > 0, 2 * p * r / (p + r + 1e-12), 0.0); b = int(np.argmax(f1))
    tpr = np.concatenate([[0.0], tp / n_ood]); fpr = np.concatenate([[0.0], (k - tp) / n_known])
    return {'best_f1': float(f1[b]), 'best_precision': float(p[b]), 'best_recall': float(r[b]),
            'norm_effort': float((b + 1) / n_ood), 'fpr': float((k[b] - tp[b]) / n_known),
            'fnr': float(1 - r[b]), 'roc_auc': float(np.trapezoid(tpr, fpr)),
            'pr_auc': float(np.trapezoid(np.concatenate([[1.0], p]), np.concatenate([[0.0], r]))),
            'n_ood': n_ood, 'n_known': int(n_known), 'n_total': int(len(is_ood))}


def mad_fixed(mad, is_ood, T=MAD_THRESHOLD):
    pred = mad > T; act = is_ood.astype(bool)
    tp = int((pred & act).sum()); fp = int((pred & ~act).sum()); fn = int((~pred & act).sum())
    tn = int((~pred & ~act).sum())
    p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {'threshold': T, 'best_f1': round(f1, 4), 'best_precision': round(p, 4), 'best_recall': round(r, 4),
            'fpr': round(fp / (fp + tn), 4) if fp + tn else 0.0, 'fnr': round(fn / (tp + fn), 4) if tp + fn else 0.0,
            'norm_effort': round((tp + fp) / (tp + fn), 4) if tp + fn else 0.0, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}


# ======================================================================
# RUN ONE TASK
# ======================================================================
def load_split(data_dir, prefix, s):
    d = np.load(os.path.join(data_dir, f'{prefix}_{s}.npz'), allow_pickle=True)
    def dense_arr(a):
        return a.item().toarray().astype(np.float32) if a.shape == () else a.astype(np.float32)
    hf = d['heldout_family']
    fam = str(hf.item()) if hf.shape == () else str(hf)
    return dense_arr(d['X_train']), d['y_train'].astype(np.int64), dense_arr(d['X_test']), d['y_test'].astype(np.int64), fam


def run_task(task_id, data_dir, prefix, result_dir, protocol, max_epochs):
    run_id, split_id = task_id // N_SPLITS, task_id % N_SPLITS
    seed = SEEDS[run_id]
    X_tr, y_tr_orig, X_te, y_te, fam = load_split(data_dir, prefix, split_id)
    known = sorted(int(c) for c in np.unique(y_tr_orig))
    remap = {c: i for i, c in enumerate(known)}                 # CADE requires labels 0..N-1
    y_tr = np.array([remap[int(c)] for c in y_tr_orig], dtype=np.int64)
    is_ood = (~np.isin(y_te, known)).astype(int)
    print('=' * 72, flush=True)
    print(f'Group16 CADE original (no PCA, 4-class) | Task={task_id} | Run={run_id} | Split={split_id} | '
          f'Seed={seed} | OOD={fam}', flush=True)
    print('=' * 72, flush=True)
    print(f'  X_train={X_tr.shape}  X_test={X_te.shape}  known labels {known} -> 0..{len(known) - 1}', flush=True)
    save_dir = os.path.join(result_dir, f'run{run_id}', f'split{split_id}')
    os.makedirs(save_dir, exist_ok=True)

    set_seed(seed)
    model = CADEAutoencoder(X_tr.shape[1], len(known))
    t0 = time.time()
    model, epochs_run, stopped_early, best_epoch, best_loss = train_cade(model, X_tr, y_tr, seed, protocol, max_epochs)
    train_time = time.time() - t0
    torch.save(model.state_dict(), os.path.join(save_dir, 'model_weights.pt'))

    model.eval()
    with torch.no_grad():
        z_tr = model.encoder(torch.from_numpy(X_tr)).numpy()
        z_te = model.encoder(torch.from_numpy(X_te)).numpy()
    np.savez_compressed(os.path.join(save_dir, 'latent.npz'), z_train=z_tr, z_test=z_te,
                        y_train=y_tr_orig, y_test=y_te)
    raw, mad, stats = cade_detect(z_tr, y_tr, z_te)

    cade_35 = cade_protocol(raw, mad, is_ood, MAD_THRESHOLD)
    cade_0  = cade_protocol(raw, mad, is_ood, 0.0)
    tf_res  = threshold_free(raw, is_ood)
    mad_res = mad_fixed(mad, is_ood)
    roc_mad = float(roc_auc_score(is_ood, mad))
    cents = [X_tr[y_tr == i].mean(axis=0) for i in range(len(known))]
    base  = float(roc_auc_score(is_ood, np.stack([np.linalg.norm(X_te - c, axis=1) for c in cents], 1).min(1)))

    print(f'  Training time: {train_time:.1f}s ({epochs_run} epochs, best checkpoint epoch {best_epoch})', flush=True)
    print(f'  [CADE protocol, T=3.5] F1={cade_35["best_f1"]:.4f}  P={cade_35["best_precision"]:.4f}  '
          f'R={cade_35["best_recall"]:.4f}  effort={cade_35["norm_effort"]:.3f}  flagged={cade_35["n_flagged"]}', flush=True)
    print(f'  [CADE protocol, T=0]   F1={cade_0["best_f1"]:.4f}', flush=True)
    print(f'  [ThreshFree raw] ROC={tf_res["roc_auc"]:.4f}  PR={tf_res["pr_auc"]:.4f}  | ROC(MAD score)={roc_mad:.4f}  '
          f'| fixed MAD F1={mad_res["best_f1"]:.4f}', flush=True)
    print(f'  [Baseline, 0 params] raw-feature centroid ROC={base:.4f}', flush=True)

    summary = {
        'model': 'Group16_CADE_Original_4class', 'reference': 'Yang et al., USENIX Security 2021 (released code)',
        'architecture': '-'.join(map(str, model.dims)) + ' linear bottleneck; mirror decoder, linear output; no PCA',
        'loss': f'{LAMBDA_1}*contrastive(margin={MARGIN}) + MSE', 'protocol': protocol,
        'run_id': run_id, 'split_id': split_id, 'ood_family': fam, 'seed': seed,
        'n_train': len(y_tr), 'n_test': len(y_te), 'n_features': X_tr.shape[1],
        'n_trainable_params': int(sum(p.numel() for p in model.parameters())),
        'lr': CAE_LR, 'max_epochs': max_epochs, 'actual_epochs': epochs_run, 'stopped_early': stopped_early,
        'best_epoch': best_epoch, 'best_loss': best_loss, 'train_time_sec': round(train_time, 2),
        'mad_stats': stats, 'roc_auc_mad_score': roc_mad, 'baseline_raw_feature_centroid_roc': base,
        'CADE_protocol_T3.5': cade_35, 'CADE_protocol_T0': cade_0,
        'MAD_Threshold': mad_res, 'Threshold_Free': tf_res,
    }
    with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(save_dir, 'ood_scores.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['raw_score', 'mad_score'])
        for a, b in zip(raw, mad):
            w.writerow([a, b])
    print(f'  Saved: {save_dir}', flush=True)


# ======================================================================
# AGGREGATE
# ======================================================================
def aggregate(result_dir):
    lines = []
    def log(s=''):
        print(s, flush=True); lines.append(s)
    rows, missing = [], []
    for r in range(3):
        for s in range(N_SPLITS):
            p = os.path.join(result_dir, f'run{r}', f'split{s}', 'summary.json')
            if not os.path.exists(p):
                missing.append((r, s)); continue
            d = json.load(open(p))
            rows.append((s, d['ood_family'], d['CADE_protocol_T3.5']['best_f1'], d['CADE_protocol_T3.5']['best_precision'],
                         d['CADE_protocol_T3.5']['best_recall'], d['CADE_protocol_T0']['best_f1'], d['MAD_Threshold']['best_f1'],
                         d['Threshold_Free']['roc_auc'], d['Threshold_Free']['pr_auc'], d['roc_auc_mad_score'],
                         d['baseline_raw_feature_centroid_roc'], d['best_epoch']))
    log('=' * 110)
    log(f'Group16 CADE original (no PCA, 4-class) -- AGGREGATED  ({result_dir})')
    log('=' * 110)
    log(f'Completed: {len(rows)}/{3 * N_SPLITS}   Missing: {missing if missing else "none"}')
    cols = ['CADE_F1@3.5', 'CADE_P@3.5', 'CADE_R@3.5', 'CADE_F1@0', 'fixedMAD_F1', 'ROC_raw', 'PR_raw',
            'ROC_mad', 'RawFeat_ROC', 'best_ep']
    log(f"  {'family':<14}{'n':>3}" + ''.join(f'{c:>15}' for c in cols))
    for s, fam in sorted({(x[0], x[1]) for x in rows}):
        A = np.array([x[2:] for x in rows if x[0] == s], float)
        log(f'  {fam:<14}{len(A):>3}' + ''.join(f'{m:>8.3f}+-{sd:<5.3f}' for m, sd in zip(A.mean(0), A.std(0))))
    if rows:
        A = np.array([x[2:] for x in rows], float)
        log(f"  {'ALL':<14}{len(A):>3}" + ''.join(f'{m:>8.3f}+-{sd:<5.3f}' for m, sd in zip(A.mean(0), A.std(0))))
    log('  CADE_F1@3.5 = CADE Table-3 metric (flag A>3.5, rank by distance, best F1). Flag-all F1 floor = 0.769.')
    log('  References (4-class LAMDA): PCA32-only ROC 0.796 | G15 MAD-score ROC 0.773 | G18 (seed 42) MAD-score ROC 0.785')
    with open(os.path.join(result_dir, 'aggregate_summary.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    p = argparse.ArgumentParser(description='Group16: CADE original (no PCA), 4-class')
    p.add_argument('task_id', help="0-11, or 'aggregate'")
    p.add_argument('--data-dir', default=DEFAULT_DATA_DIR)
    p.add_argument('--prefix', default='drebin_new')
    p.add_argument('--results-dir', default='results')
    p.add_argument('--protocol', choices=['cade', 'matched'], default='cade',
                   help="'cade' = CADE's released training/checkpointing; 'matched' = Groups 14/17/18 rule")
    p.add_argument('--max-epochs', type=int, default=CAE_EPOCHS)
    a = p.parse_args()
    if a.task_id == 'aggregate':
        aggregate(os.path.expanduser(a.results_dir))
    else:
        tid = int(a.task_id)
        if not 0 <= tid < 3 * N_SPLITS:
            raise ValueError(f'task_id must be 0-{3 * N_SPLITS - 1}')
        run_task(tid, os.path.expanduser(a.data_dir), a.prefix, os.path.expanduser(a.results_dir),
                 a.protocol, a.max_epochs)


if __name__ == '__main__':
    main()
