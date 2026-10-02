#!/usr/bin/env python3
"""
run_quantum_angle_8.py -- capacity_matched_8fam/pca8 / quantum_angle_8   (standalone)
Quantum, ANGLE encoding: the Quantum v2 design (selected on validation at PCA-5) on 8 qubits, one PCA feature per qubit.
  input    PCA-8 -> center, divide all 8 by ONE shared scale (std of PC1) -> arctan
  circuit  8 qubits, 8 trainable layers (RY, RZ on every qubit + CNOT ring + CNOT 0->2, 2->4, 4->6 on every second
           layer); the data is RE-UPLOADED 4 times with trainable input scaling RY(a_k * x_k + b_k)
  params   8 x 8 x 2 = 128 circuit + 4 x 8 x 2 = 64 input scaling = 192
  loss     CADE's contrastive form on d = 1 - |<psi|psi'>|^2, margin 1, CADE pairs
  scoring  mixed density-matrix prototypes, d_c = 1 - <psi|rho_c|psi>, + CADE's MAD
Experiment: 8-family LAMDA (cade_lamda), PCA-8 input, capacity-matched (quantum 192 params; each CADE-mini
version as close as its layer widths allow). 24 tasks = 8 held-out families x 3 seeds; 7 known families per task, so the
CADE-mini bottleneck is 7-dimensional (CADE: latent size = number of known families). Shared: CADE pairing (batch 64), Adam LR 0.001, early
stopping (patience 20, rel 1e-4, max 250, best weights), MAD detection, metrics, threshold sweep, per-task and aggregate
plots, ood_scores.csv + train scores for evaluate.py / compare_all.py.
Usage (from capacity_matched_8fam/pca8/quantum_angle_8/):
  python3 run_quantum_angle_8.py probe                          # 1-2 min, no training
  mkdir -p logs && sbatch run_validate.slurm                    # 12 validation jobs (threshold selection)
  mkdir -p logs && sbatch run_quantum_angle_8.slurm             # 24 test tasks
  python3 run_quantum_angle_8.py aggregate
"""

# =============================================================================
# SHARED CODE -- identical in all capacity_matched scripts of this experiment
# (data, PCA-5, CADE pairing, early stopping, MAD detection, metrics,
#  threshold sweep, saving, plots, aggregation). Only the MODEL section differs.
# =============================================================================


import os, sys, csv, json, time, copy
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

# ----------------------------------------------------------------------------- config
SEEDS          = {0: 42, 1: 123, 2: 456}
N_SPLITS       = 8
N_TASKS        = 3 * N_SPLITS
PCA_DIM        = 8               # PCA-8 input (quantum: 8 qubits, one feature per qubit)
THRESHOLDS     = [0.0, 2.0, 2.5, 3.0, 3.5]
MAD_T          = 3.5
MAD_B          = 1.4826
BATCH_SIZE     = 64
SIMILAR_RATIO  = 0.25
MAX_EPOCHS     = 250
PATIENCE       = 20
REL_MIN_DELTA  = 1e-4
LR             = 0.001
DEFAULT_DATA   = '~/projects/OOD/Data/cade_lamda'
DEFAULT_PREFIX = 'drebin_new'
FLAG_ALL_F1    = None           # flag-all F1 floor: computed per task from its test set (2p/(1+p), p = OOD share)


def set_seed(seed):
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def task_info(task_id):
    run, split = task_id // N_SPLITS, task_id % N_SPLITS
    return run, split, SEEDS[run]


# ----------------------------------------------------------------------------- data
def load_split(data_dir, prefix, s):
    d = np.load(os.path.join(os.path.expanduser(data_dir), f'{prefix}_{s}.npz'), allow_pickle=True)
    def dense(a):
        return a.item().toarray().astype(np.float32) if a.shape == () else a.astype(np.float32)
    hf = d['heldout_family']
    fam = str(hf.item()) if hf.shape == () else str(hf)
    Xtr, ytr, Xte, yte = dense(d['X_train']), d['y_train'].astype(np.int64), dense(d['X_test']), d['y_test'].astype(np.int64)
    known = sorted(int(c) for c in np.unique(ytr))
    remap = {c: i for i, c in enumerate(known)}
    ytr_c = np.array([remap[int(c)] for c in ytr], dtype=np.int64)          # 0..N-1, as CADE requires
    is_ood = (~np.isin(yte, known)).astype(int)
    return {'Xtr': Xtr, 'ytr': ytr_c, 'Xte': Xte, 'yte': yte, 'is_ood': is_ood, 'family': fam,
            'known_labels': known}


def fit_pca(Xtr):
    return PCA(n_components=PCA_DIM, svd_solver='full').fit(Xtr)


# ----------------------------------------------------------------------------- CADE pairing
def epoch_batches(X, y, batch_size=BATCH_SIZE, similar_ratio=SIMILAR_RATIO):
    """CADE's data.epoch_batches (github.com/whyisyoung/CADE). y must be 0..N-1."""
    half = batch_size // 2
    bc = int(X.shape[0] / half)
    num_sim = int(batch_size * similar_ratio)
    bx = np.zeros([bc, batch_size, X.shape[1]], dtype=np.float32)
    by = np.zeros([bc, batch_size], dtype=np.int64)
    idx = np.random.permutation(X.shape[0])
    bx[:, :half] = np.split(X[idx[: bc * half]], bc)
    by[:, :half] = np.split(y[idx[: bc * half]], bc)
    cls = [np.where(y == c)[0] for c in range(len(np.unique(y)))]
    ncls = [np.where(y != c)[0] for c in range(len(np.unique(y)))]
    for b in range(bc):
        for m in range(num_sim):
            j = np.random.choice(cls[by[b, m]], 1)[0]
            bx[b, m + half], by[b, m + half] = X[j], y[j]
        for m in range(num_sim, half):
            j = np.random.choice(ncls[by[b, m]], 1)[0]
            bx[b, m + half], by[b, m + half] = X[j], y[j]
    return bc, bx, by


# ----------------------------------------------------------------------------- stopping
class EarlyStopper:
    """Per-epoch training loss; stop after PATIENCE epochs without a relative improvement
    of REL_MIN_DELTA; remember the best state (restored by the caller)."""
    def __init__(self):
        self.best, self.best_epoch, self.since, self.state = np.inf, 0, 0, None

    def step(self, epoch, loss, state_fn):
        if loss < self.best * (1.0 - REL_MIN_DELTA):
            self.best, self.best_epoch, self.since = loss, epoch, 0
            self.state = copy.deepcopy(state_fn())
        else:
            self.since += 1
        return self.since >= PATIENCE


# ----------------------------------------------------------------------------- detection
def _mad_stats(d):
    med = float(np.median(d))
    mad = float(MAD_B * np.median(np.abs(d - med)))
    return med, (mad if mad > 0 else 1e-12)


def centroid_detect(Ztr, ytr, Zte):
    """Euclidean distance to known-class mean centroids (CADE detect.py)."""
    D, A = [], []
    for c in range(len(np.unique(ytr))):
        mu = Ztr[ytr == c].mean(axis=0)
        med, mad = _mad_stats(np.linalg.norm(Ztr[ytr == c] - mu, axis=1))
        dt = np.linalg.norm(Zte - mu, axis=1)
        D.append(dt); A.append(np.abs(dt - med) / mad)
    return np.min(D, axis=0), np.min(A, axis=0)


def prototype_detect(Str, ytr, Ste):
    """Quantum analogue: d_c = 1 - <psi|rho_c|psi>, rho_c = mean training state of class c
    (leave-one-out for training samples)."""
    D, A = [], []
    for c in range(len(np.unique(ytr))):
        P = Str[ytr == c]; n = len(P)
        F_tr = np.abs(P.conj() @ P.T) ** 2
        med, mad = _mad_stats(1.0 - (F_tr.sum(axis=1) - 1.0) / (n - 1))
        dt = 1.0 - (np.abs(Ste.conj() @ P.T) ** 2).mean(axis=1)
        D.append(dt); A.append(np.abs(dt - med) / mad)
    return np.min(D, axis=0), np.min(A, axis=0)


# ----------------------------------------------------------------------------- evaluation
def _best_down_list(raw, is_ood, idx):
    n_ood, n_known = int(is_ood.sum()), int(len(is_ood) - is_ood.sum())
    if len(idx) == 0:
        return {'F1': 0.0, 'Precision': 0.0, 'Recall': 0.0, 'FPR': 0.0, 'FNR': 1.0, 'Effort': 0.0, 'k': 0}
    order = idx[np.argsort(-raw[idx], kind='stable')]
    tp = np.cumsum(is_ood[order]); k = np.arange(1, len(order) + 1)
    p, r = tp / k, tp / n_ood
    f = np.where(p + r > 0, 2 * p * r / (p + r + 1e-12), 0.0); b = int(np.argmax(f))
    return {'F1': float(f[b]), 'Precision': float(p[b]), 'Recall': float(r[b]),
            'FPR': float((k[b] - tp[b]) / n_known), 'FNR': float(1 - r[b]),
            'Effort': float(k[b] / n_ood), 'k': int(k[b])}


def evaluate(raw, mad, is_ood):
    res = {'threshold_free': _best_down_list(raw, is_ood, np.arange(len(raw)))}
    res['threshold_free']['ROC_AUC'] = float(roc_auc_score(is_ood, raw))
    res['threshold_free']['PR_AUC'] = float(average_precision_score(is_ood, raw))
    res['ROC_AUC_mad'] = float(roc_auc_score(is_ood, mad))
    res['PR_AUC_mad'] = float(average_precision_score(is_ood, mad))
    pred = mad > MAD_T; act = is_ood.astype(bool)
    tp, fp, fn = int((pred & act).sum()), int((pred & ~act).sum()), int((~pred & act).sum())
    P = tp / (tp + fp) if tp + fp else 0.0; R = tp / (tp + fn) if tp + fn else 0.0
    res['mad_fixed'] = {'T': MAD_T, 'F1': (2 * P * R / (P + R) if P + R else 0.0), 'Precision': P, 'Recall': R,
                        'FPR': fp / (len(act) - act.sum()), 'Frac_flagged': float(pred.mean()), 'TP': tp, 'FP': fp}
    res['sweep'] = []
    for T in THRESHOLDS:
        idx = np.arange(len(raw)) if T == 0 else np.where(mad > T)[0]
        e = _best_down_list(raw, is_ood, idx); e['T_MAD'] = T; e['Frac_flagged'] = float(len(idx) / len(raw))
        res['sweep'].append(e)
    return res


# ----------------------------------------------------------------------------- saving + plots
def save_task(save_dir, model_name, meta, raw, mad, data, loss_hist, lat_tr2d, lat_te2d, lat_kind):
    os.makedirs(os.path.join(save_dir, 'plots'), exist_ok=True)
    ev = evaluate(raw, mad, data['is_ood'])
    summary = {'model': model_name, **meta, 'ood_family': data['family'],
               'n_train': int(len(data['ytr'])), 'n_test': int(len(data['yte'])),
               'n_ood': int(data['is_ood'].sum()), 'n_known_test': int(len(data['is_ood']) - data['is_ood'].sum()),
               'evaluation': ev}
    with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    with open(os.path.join(save_dir, 'ood_scores.csv'), 'w') as f:
        f.write('raw_score,mad_score,is_ood,true_label\n')
        for a, b, o, y in zip(raw, mad, data['is_ood'], data['yte']):
            f.write(f'{a},{b},{o},{y}\n')
    with open(os.path.join(save_dir, 'sweep.csv'), 'w') as f:
        keys = ['T_MAD', 'F1', 'Precision', 'Recall', 'FPR', 'FNR', 'Effort', 'Frac_flagged', 'k']
        f.write(','.join(keys) + '\n')
        for e in ev['sweep']:
            f.write(','.join(str(e[k]) for k in keys) + '\n')
    if loss_hist:
        with open(os.path.join(save_dir, 'training_loss.csv'), 'w') as f:
            f.write('epoch,loss\n')
            for i, l in enumerate(loss_hist):
                f.write(f'{i + 1},{l}\n')
    plot_task(os.path.join(save_dir, 'plots'), model_name, data, raw, mad, ev, loss_hist,
              meta.get('best_epoch'), lat_tr2d, lat_te2d, lat_kind)
    return summary


def plot_task(pdir, model_name, data, raw, mad, ev, loss_hist, best_epoch, lat_tr2d, lat_te2d, lat_kind):
    o = data['is_ood']; fam = data['family']; tag = f'{model_name} | OOD = {fam}'
    # 1. ROC + PR
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for s, lab in [(raw, 'raw distance'), (mad, 'MAD score')]:
        fpr, tpr, _ = roc_curve(o, s); ax[0].plot(fpr, tpr, label=f'{lab} (AUC {roc_auc_score(o, s):.3f})')
        p, r, _ = precision_recall_curve(o, s); ax[1].plot(r, p, label=f'{lab} (AP {average_precision_score(o, s):.3f})')
    ax[0].plot([0, 1], [0, 1], 'k:', lw=1); ax[0].set_xlabel('FPR'); ax[0].set_ylabel('TPR'); ax[0].set_title('ROC')
    ax[1].axhline(o.mean(), color='k', ls=':', lw=1, label=f'prevalence {o.mean():.3f}')
    ax[1].set_xlabel('Recall'); ax[1].set_ylabel('Precision'); ax[1].set_title('Precision-Recall')
    for a in ax:
        a.legend(fontsize=8); a.grid(alpha=0.3)
    fig.suptitle(tag); fig.tight_layout(); fig.savefig(os.path.join(pdir, 'roc_pr.png'), dpi=130); plt.close(fig)
    # 2. CADE Fig.4-style inspection curves (T=0 and T=3.5)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for a, T in zip(ax, [0.0, MAD_T]):
        idx = np.arange(len(raw)) if T == 0 else np.where(mad > T)[0]
        if len(idx):
            order = idx[np.argsort(-raw[idx], kind='stable')]
            tp = np.cumsum(o[order]); k = np.arange(1, len(order) + 1)
            a.plot(k, tp / k, label='precision'); a.plot(k, tp / o.sum(), '--', label='recall')
        e = [x for x in ev['sweep'] if x['T_MAD'] == T][0]
        a.set_title(f'T_MAD={T}: flagged {len(idx)}, best F1 {e["F1"]:.3f}')
        a.set_xlabel('# inspected samples (ranked by distance)'); a.set_ylim(0, 1.05); a.grid(alpha=0.3)
        if len(idx):
            a.legend(fontsize=8)
        else:
            a.text(0.5, 0.5, 'no sample exceeds T_MAD', ha='center', transform=a.transAxes)
    fig.suptitle(tag); fig.tight_layout(); fig.savefig(os.path.join(pdir, 'inspection_curves.png'), dpi=130); plt.close(fig)
    # 3. score distributions
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, s, lab in [(ax[0], raw, 'raw distance to nearest class'), (ax[1], mad, 'MAD anomaly score')]:
        bins = np.linspace(np.percentile(s, 0.5), np.percentile(s, 99.5), 50)
        a.hist(s[o == 0], bins=bins, alpha=0.6, label='known (ID)', density=True)
        a.hist(s[o == 1], bins=bins, alpha=0.6, label=f'OOD ({fam})', density=True)
        a.set_xlabel(lab); a.legend(fontsize=8); a.grid(alpha=0.3)
    ax[1].axvline(MAD_T, color='r', ls='--', lw=1)
    fig.suptitle(tag); fig.tight_layout(); fig.savefig(os.path.join(pdir, 'score_distributions.png'), dpi=130); plt.close(fig)
    # 4. training loss
    if loss_hist:
        fig, a = plt.subplots(figsize=(6, 4))
        a.plot(range(1, len(loss_hist) + 1), loss_hist, marker='.')
        if best_epoch:
            a.axvline(best_epoch, color='r', ls='--', lw=1, label=f'best epoch {best_epoch}')
        a.set_xlabel('epoch'); a.set_ylabel('training loss'); a.set_title(tag, fontsize=9); a.grid(alpha=0.3); a.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(pdir, 'training_loss.png'), dpi=130); plt.close(fig)
    # 5. latent space (2-D projection)
    if lat_te2d is not None:
        fig, a = plt.subplots(figsize=(6.5, 5.5))
        if lat_tr2d is not None:
            for c in np.unique(data['ytr']):
                m = data['ytr'] == c
                a.scatter(lat_tr2d[m, 0], lat_tr2d[m, 1], s=4, alpha=0.25, label=f'train class {data["known_labels"][c]}')
        a.scatter(lat_te2d[o == 1, 0], lat_te2d[o == 1, 1], s=8, marker='x', c='k', alpha=0.6, label=f'test OOD ({fam})')
        a.set_title(f'{tag}\nlatent: {lat_kind}', fontsize=9); a.legend(fontsize=7, markerscale=2); a.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(pdir, 'latent_space.png'), dpi=130); plt.close(fig)


# ----------------------------------------------------------------------------- aggregation
HEADLINE = [('TF_F1', lambda e: e['threshold_free']['F1']),
            ('TF_Precision', lambda e: e['threshold_free']['Precision']),
            ('TF_Recall', lambda e: e['threshold_free']['Recall']),
            ('ROC_raw', lambda e: e['threshold_free']['ROC_AUC']),
            ('PR_raw', lambda e: e['threshold_free']['PR_AUC']),
            ('ROC_mad', lambda e: e['ROC_AUC_mad']),
            ('MAD_F1_fixed', lambda e: e['mad_fixed']['F1'])] + \
           [(f'F1@T{T}', (lambda T: lambda e: [x for x in e['sweep'] if x['T_MAD'] == T][0]['F1'])(T)) for T in THRESHOLDS] + \
           [(f'Flagged@T{T}', (lambda T: lambda e: [x for x in e['sweep'] if x['T_MAD'] == T][0]['Frac_flagged'])(T)) for T in THRESHOLDS[1:]]


def collect(results_dir):
    rows = []
    for r in range(3):
        for s in range(N_SPLITS):
            p = os.path.join(results_dir, f'run{r}', f'split{s}', 'summary.json')
            if os.path.exists(p):
                d = json.load(open(p))
                p_ood = d.get('n_ood', 0) / max(d.get('n_test', 1), 1)
                row = {'run': r, 'split': s, 'family': d['ood_family'], 'n_params': d.get('n_trainable_params', 0),
                       'best_epoch': d.get('best_epoch', 0), 'floor': 2 * p_ood / (1 + p_ood)}
                row.update({k: f(d['evaluation']) for k, f in HEADLINE})
                rows.append(row)
    return rows


def aggregate(results_dir, model_name):
    rows = collect(results_dir)
    lines = []
    def log(s=''):
        print(s, flush=True); lines.append(s)
    missing = [(r, s) for r in range(3) for s in range(N_SPLITS) if not any(x['run'] == r and x['split'] == s for x in rows)]
    log('=' * 110); log(f'{model_name} -- AGGREGATED  ({results_dir})'); log('=' * 110)
    log(f'Completed: {len(rows)}/{N_TASKS}   Missing: {missing if missing else "none"}')
    if not rows:
        return
    keys = [k for k, _ in HEADLINE]
    fams = []
    for x in rows:
        if x['family'] not in fams:
            fams.append(x['family'])
    show = ['TF_F1', 'ROC_raw', 'PR_raw', 'ROC_mad', 'MAD_F1_fixed', 'F1@T3.5', 'Flagged@T3.5']
    log(f"  {'family':<13}{'n':>3}" + ''.join(f'{k:>16}' for k in show))
    for fam in fams + ['ALL']:
        sel = rows if fam == 'ALL' else [x for x in rows if x['family'] == fam]
        log(f'  {fam:<13}{len(sel):>3}' + ''.join(
            f'{np.mean([x[k] for x in sel]):>9.3f}+-{np.std([x[k] for x in sel]):<5.3f}' for k in show))
    log(f'\n  Threshold sweep (CADE protocol, all tasks):')
    for T in THRESHOLDS:
        f1 = [x[f'F1@T{T}'] for x in rows]
        fl = [x[f'Flagged@T{T}'] for x in rows] if T > 0 else [1.0]
        log(f'    T_MAD={T:<4} F1={np.mean(f1):.3f}+-{np.std(f1):.3f}   flagged={np.mean(fl):.3f}')
    floor = float(np.mean([x['floor'] for x in rows]))
    log(f'  Threshold-free F1 flag-all floor = {floor:.3f} (mean over tasks). std = population std.')
    log(f'  Params: {rows[0]["n_params"]}   best epochs: {[x["best_epoch"] for x in rows]}')
    with open(os.path.join(results_dir, 'aggregate_summary.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    with open(os.path.join(results_dir, 'aggregate.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['run', 'split', 'family', 'n_params', 'best_epoch', 'floor'] + keys)
        w.writeheader(); w.writerows(rows)
    # plots
    pdir = os.path.join(results_dir, 'aggregate_plots'); os.makedirs(pdir, exist_ok=True)
    fig, a = plt.subplots(figsize=(6, 4))
    m = [np.mean([x[f'F1@T{T}'] for x in rows]) for T in THRESHOLDS]; sd = [np.std([x[f'F1@T{T}'] for x in rows]) for T in THRESHOLDS]
    a.errorbar(THRESHOLDS, m, yerr=sd, marker='o', capsize=3)
    a.axhline(floor, color='k', ls=':', lw=1, label='flag-all F1 (T=0 floor)')
    a.set_xlabel('T_MAD'); a.set_ylabel('best F1 (CADE protocol)'); a.set_title(f'{model_name}: threshold sweep'); a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(pdir, 'threshold_sweep.png'), dpi=130); plt.close(fig)
    fig, a = plt.subplots(figsize=(8, 4.2)); w = 0.2; x0 = np.arange(len(fams))
    for i, k in enumerate(['ROC_raw', 'ROC_mad', 'TF_F1', 'MAD_F1_fixed']):
        mm = [np.mean([x[k] for x in rows if x['family'] == f]) for f in fams]
        ss = [np.std([x[k] for x in rows if x['family'] == f]) for f in fams]
        a.bar(x0 + (i - 1.5) * w, mm, w, yerr=ss, capsize=2, label=k)
    a.set_xticks(x0); a.set_xticklabels(fams); a.set_ylim(0, 1.05); a.grid(alpha=0.3, axis='y'); a.legend(fontsize=8)
    a.set_title(f'{model_name}: per family (mean +- std over runs)')
    fig.tight_layout(); fig.savefig(os.path.join(pdir, 'per_family.png'), dpi=130); plt.close(fig)
    print(f'\nSaved: {results_dir}/aggregate_summary.txt, aggregate.csv, aggregate_plots/*.png')


# ----------------------------------------------------------------------------- classical CADE-mini (shared by 2 folders)
def run_cade_mini(task_id, data_dir, prefix, results_dir, hidden, reconstruction, model_name, max_epochs=MAX_EPOCHS, margin=10.0):
    """CADE's contrastive autoencoder (released code), scaled down, on PCA-5.
    Linear bottleneck of size N_known = 3; ReLU hidden; Glorot init; margin 10;
    loss = 0.1*contrastive + MSE (reconstruction=True) or contrastive only (False)."""
    import torch, torch.nn as nn, torch.optim as optim
    run, split, seed = task_info(task_id)
    data = load_split(data_dir, prefix, split)
    pca = fit_pca(data['Xtr'])
    Ztr = pca.transform(data['Xtr']).astype(np.float32); Zte = pca.transform(data['Xte']).astype(np.float32)
    n_lat = len(np.unique(data['ytr']))
    set_seed(seed)

    def dense(i, o):
        l = nn.Linear(i, o); nn.init.xavier_uniform_(l.weight); nn.init.zeros_(l.bias); return l
    dims = [PCA_DIM] + list(hidden) + [n_lat]
    enc = []
    for i in range(len(dims) - 2):
        enc += [dense(dims[i], dims[i + 1]), nn.ReLU()]
    enc += [dense(dims[-2], dims[-1])]
    encoder = nn.Sequential(*enc)
    decoder = None
    if reconstruction:
        dec = []
        for i in range(len(dims) - 1, 1, -1):
            dec += [dense(dims[i], dims[i - 1]), nn.ReLU()]
        dec += [dense(dims[1], dims[0])]
        decoder = nn.Sequential(*dec)
    modules = nn.ModuleList([encoder] + ([decoder] if decoder is not None else []))
    n_params = sum(p.numel() for p in modules.parameters())
    opt = optim.Adam(modules.parameters(), lr=LR, eps=1e-7)
    print(f'{model_name} | task {task_id} run {run} split {split} seed {seed} OOD={data["family"]}', flush=True)
    print(f'  encoder {"-".join(map(str, dims))} (linear bottleneck), decoder: {"mirror" if decoder is not None else "none"}, '
          f'params {n_params}, loss {"0.1*contrastive + MSE" if reconstruction else "contrastive only"}', flush=True)

    stop, hist, t0 = EarlyStopper(), [], time.time()
    for epoch in range(1, max_epochs + 1):
        bc, bx, by = epoch_batches(Ztr, data['ytr'])
        ls = []
        for b in range(bc):
            xb, yb = torch.from_numpy(bx[b]), torch.from_numpy(by[b])
            opt.zero_grad()
            z = encoder(xb); h = z.shape[0] // 2
            dist = torch.sqrt(((z[:h] - z[h:]) ** 2).sum(1) + 1e-10)
            same = (yb[:h] == yb[h:]).float()
            con = (same * dist + (1 - same) * torch.relu(margin - dist)).mean()
            loss = 0.1 * con + ((decoder(z) - xb) ** 2).mean(1).mean() if decoder is not None else con
            loss.backward(); opt.step(); ls.append(loss.item())
        hist.append(float(np.mean(ls)))
        done = stop.step(epoch, hist[-1], modules.state_dict)
        print(f'  epoch {epoch:3d} loss {hist[-1]:.5f} best {stop.best:.5f} (ep {stop.best_epoch}) no_improve {stop.since}', flush=True)
        if done:
            break
    modules.load_state_dict(stop.state)
    with torch.no_grad():
        Ltr = encoder(torch.from_numpy(Ztr)).numpy(); Lte = encoder(torch.from_numpy(Zte)).numpy()
    raw, mad = centroid_detect(Ltr, data['ytr'], Lte)
    p2 = PCA(n_components=2).fit(Ltr) if Ltr.shape[1] > 2 else None
    lat_tr = p2.transform(Ltr) if p2 else Ltr; lat_te = p2.transform(Lte) if p2 else Lte
    save_dir = os.path.join(results_dir, f'run{run}', f'split{split}')
    meta = {'run_id': run, 'split_id': split, 'seed': seed, 'input': f'PCA-{PCA_DIM}',
            'pca_explained_variance': float(pca.explained_variance_ratio_.sum()),
            'architecture': f'encoder {"-".join(map(str, dims))}, decoder {"mirror" if decoder is not None else "none"}',
            'reconstruction': reconstruction, 'n_trainable_params': int(n_params), 'lr': LR,
            'epochs_run': len(hist), 'best_epoch': stop.best_epoch, 'best_loss': stop.best,
            'patience': PATIENCE, 'train_time_sec': round(time.time() - t0, 1)}
    os.makedirs(save_dir, exist_ok=True)
    torch.save(modules.state_dict(), os.path.join(save_dir, 'model_weights.pt'))
    np.save(os.path.join(save_dir, 'train_anomaly_scores.npy'), centroid_detect(Ltr, data['ytr'], Ltr)[1])
    s = save_task(save_dir, model_name, meta, raw, mad, data, hist, lat_tr, lat_te, '2-D PCA of the 3-D latent vectors')
    e = s['evaluation']
    print(f'  DONE  ROC_raw {e["threshold_free"]["ROC_AUC"]:.4f}  ROC_mad {e["ROC_AUC_mad"]:.4f}  '
          f'TF_F1 {e["threshold_free"]["F1"]:.4f}  MAD_F1 {e["mad_fixed"]["F1"]:.4f}  -> {save_dir}', flush=True)


def standard_main(description, run_fn, model_name):
    import argparse
    p = argparse.ArgumentParser(description=description)
    p.add_argument('task_id', help=f"0-{N_TASKS - 1}, 'all', or 'aggregate'")
    p.add_argument('--data-dir', default=DEFAULT_DATA)
    p.add_argument('--prefix', default=DEFAULT_PREFIX)
    p.add_argument('--results-dir', default='results')
    p.add_argument('--max-epochs', type=int, default=MAX_EPOCHS)
    a = p.parse_args()
    rd = os.path.abspath(os.path.expanduser(a.results_dir))
    if a.task_id == 'aggregate':
        aggregate(rd, model_name)
    elif a.task_id == 'all':
        for t in range(N_TASKS):
            run_fn(t, a.data_dir, a.prefix, rd, a.max_epochs)
        aggregate(rd, model_name)
    else:
        run_fn(int(a.task_id), a.data_dir, a.prefix, rd, a.max_epochs)


# =============================================================================
# MODEL-SPECIFIC CODE -- quantum v2 at PCA-8 on 8 qubits
# =============================================================================
import argparse

HERE       = os.path.dirname(os.path.abspath(__file__))
SEL_DIR    = os.path.join(HERE, 'selection')
N_LAYERS_8 = 8                      # 4 uploads -> one upload every 2 layers (as v2: before layers 1, 3, 5, 7)
N_QUBITS   = PCA_DIM
N_UPLOADS  = 4
INIT_STD   = 0.1
SEL_EPOCHS = 60                     # max epochs per selection run (same stopping rule otherwise)
SEL_SEED   = 42
N_FOLDS    = 3                      # pseudo-OOD folds per split = number of known families

# The candidate variants (fixed in advance). Every variant has ~200 trainable parameters.
# Quantum v2 exactly as selected at PCA-5 (scaled input, distance 1 - F, margin 1), scaled to 8 qubits.
VARIANTS = [{'scaling': True, 'distance': 'fid_lin', 'margin': 1.0}]
DIST_DOC = {'fid_sqrt': 'sqrt(1 - fidelity) [= v1]', 'fid_lin': '1 - fidelity',
            'local': 'local (projected): per-qubit Bloch-vector distance'}


def n_layers(v):
    return N_LAYERS_8                             # 8 layers x 8 qubits x 2 = 128 + input scaling 4 x 8 x 2 = 64 -> 192


def vname(v):
    return f"{'scaled' if v['scaling'] else 'fixed'}-input / {v['distance']} / margin {v['margin']}"


def make_circuit(L):
    import pennylane as qml
    dev = qml.device('default.qubit', wires=N_QUBITS)
    every = L // N_UPLOADS

    @qml.qnode(dev, interface='torch', diff_method='backprop')
    def circuit(angles, weights, sa, sb):
        for l in range(L):
            if l % every == 0:
                u = l // every
                for k in range(N_QUBITS):
                    qml.RY(sa[u, k] * angles[..., k] + sb[u, k], wires=k)     # (1) input scaling
            for k in range(N_QUBITS):
                qml.RY(weights[l, k, 0], wires=k)
                qml.RZ(weights[l, k, 1], wires=k)
            for k in range(N_QUBITS - 1):
                qml.CNOT(wires=[k, k + 1])
            qml.CNOT(wires=[N_QUBITS - 1, 0])
            if l % 2 == 1:
                for k in range(0, N_QUBITS - 2, 2):
                    qml.CNOT(wires=[k, k + 2])
        return qml.state()
    return circuit


def bloch_torch(psi):
    """Per-qubit Bloch vectors of a batch of pure states: [B, 2^n] -> [B, n, 3] (wire 0 = MSB)."""
    import torch
    B = psi.shape[0]; t = psi.reshape([B] + [2] * N_QUBITS); out = []
    for k in range(N_QUBITS):
        m = torch.movedim(t, k + 1, 1).reshape(B, 2, -1)
        rho = m @ m.conj().transpose(1, 2)
        out.append(torch.stack([2 * rho[:, 0, 1].real, -2 * rho[:, 0, 1].imag,
                                (rho[:, 0, 0] - rho[:, 1, 1]).real], dim=-1))
    return torch.stack(out, dim=1)


def pair_distance(sa, sb, distance):
    """(2) distance between the two halves of a batch. All three lie in [0, 1]."""
    import torch
    if distance == 'local':
        ra, rb = bloch_torch(sa), bloch_torch(sb)
        return torch.sqrt(((ra - rb) ** 2).sum(dim=(1, 2)) / (4.0 * N_QUBITS) + 1e-10)
    ov = (sa.conj() * sb).sum(-1)
    F = (ov.real ** 2 + ov.imag ** 2).clamp(0.0, 1.0)
    return torch.sqrt((1.0 - F).clamp(min=0.0) + 1e-10) if distance == 'fid_sqrt' else 1.0 - F


def train_embed(Xtr, ytr, X_eval, v, seed, max_epochs, tag=''):
    """Fit PCA-5 on Xtr, train the variant, return features for Xtr and each X in X_eval."""
    import torch, torch.optim as optim
    pca = fit_pca(Xtr)
    Ztr = pca.transform(Xtr); mu, scale = Ztr.mean(axis=0), float(Ztr[:, 0].std()) + 1e-8
    ang = lambda X: np.arctan((pca.transform(X) - mu) / scale).astype(np.float32)
    Atr = ang(Xtr)
    L = n_layers(v)
    set_seed(seed)
    weights = torch.nn.Parameter(torch.randn(L, N_QUBITS, 2) * INIT_STD)
    params = [weights]
    if v['scaling']:
        sa = torch.nn.Parameter(torch.ones(N_UPLOADS, N_QUBITS)); sb = torch.nn.Parameter(torch.zeros(N_UPLOADS, N_QUBITS))
        params += [sa, sb]
    else:
        sa, sb = torch.ones(N_UPLOADS, N_QUBITS), torch.zeros(N_UPLOADS, N_QUBITS)
    n_params = int(sum(p.numel() for p in params))
    circuit = make_circuit(L)
    opt = optim.Adam(params, lr=LR)

    def states(A):
        psi = circuit(A, weights, sa, sb)
        return psi if psi.dim() == 2 else psi.unsqueeze(0)

    print(f'  {tag}variant: {vname(v)} | {L} layers, {N_UPLOADS} uploads, {n_params} params', flush=True)
    stop, hist, t0 = EarlyStopper(), [], time.time()
    snap = lambda: {'w': weights.detach().clone(), 'a': sa.detach().clone(), 'b': sb.detach().clone()}
    stop.state = snap()
    for epoch in range(1, max_epochs + 1):
        bc, bx, by = epoch_batches(Atr, ytr)
        ls = []
        for b in range(bc):
            A, yb = torch.from_numpy(bx[b]), torch.from_numpy(by[b])
            opt.zero_grad()
            st = states(A); h = st.shape[0] // 2
            d = pair_distance(st[:h], st[h:], v['distance'])
            same = (yb[:h] == yb[h:]).float()
            loss = (same * d + (1 - same) * torch.relu(v['margin'] - d)).mean()
            loss.backward(); opt.step(); ls.append(loss.item())
        hist.append(float(np.mean(ls)))
        done = stop.step(epoch, hist[-1], snap)
        print(f'  {tag}epoch {epoch:3d} loss {hist[-1]:.5f} best {stop.best:.5f} (ep {stop.best_epoch}) '
              f'no_improve {stop.since} ({time.time() - t0:.0f}s)', flush=True)
        if done:
            break
    with torch.no_grad():
        weights.copy_(stop.state['w'])
        if v['scaling']:
            sa.copy_(stop.state['a']); sb.copy_(stop.state['b'])

    def feats(X, bs=256):
        A = ang(X); out = []
        with torch.no_grad():
            for i in range(0, len(A), bs):
                st = states(torch.from_numpy(A[i:i + bs]))
                out.append(bloch_torch(st).reshape(len(st), -1).numpy() if v['distance'] == 'local' else st.numpy())
        return np.vstack(out)
    info = {'n_params': n_params, 'epochs_run': len(hist), 'best_epoch': stop.best_epoch,
            'best_loss': (None if not np.isfinite(stop.best) else stop.best), 'hist': hist,
            'train_time_sec': round(time.time() - t0, 1), 'pca': pca, 'mu': mu, 'scale': scale,
            'weights': weights.detach(), 'sa': sa.detach(), 'sb': sb.detach()}
    return feats(Xtr), [feats(X) for X in X_eval], info


def detect(Ftr, ytr, Fev, v):
    """(2) scoring consistent with training: fidelity -> density-matrix prototypes;
    local -> Euclidean centroids of the per-qubit Bloch vectors."""
    return centroid_detect(Ftr, ytr, Fev) if v['distance'] == 'local' else prototype_detect(Ftr, ytr, Fev)


def train_anomaly(Ftr, ytr, v):
    """Anomaly scores A of the TRAINING samples (for threshold calibration in compare_all.py)."""
    if v['distance'] == 'local':
        return centroid_detect(Ftr, ytr, Ftr)[1]
    C = len(np.unique(ytr)); D = np.zeros((len(ytr), C)); own = []
    for c in range(C):
        m = ytr == c; P = Ftr[m]; n = m.sum()
        Fk = np.abs(Ftr.conj() @ P.T) ** 2
        Dc = 1.0 - Fk.mean(axis=1); Dc[m] = 1.0 - (Fk[m].sum(axis=1) - 1.0) / (n - 1)
        D[:, c] = Dc; own.append(Dc[m])
    A = []
    for c, d in enumerate(own):
        med, mad = _mad_stats(d); A.append(np.abs(D[:, c] - med) / mad)
    return np.min(A, axis=0)


def train_raw(Ftr, ytr, v):
    """Raw distances of the TRAINING samples to their nearest prototype (own class leave-one-out)."""
    if v['distance'] == 'local':
        return centroid_detect(Ftr, ytr, Ftr)[0]
    C = len(np.unique(ytr)); D = np.zeros((len(ytr), C))
    for c in range(C):
        m = ytr == c; Fk = np.abs(Ftr.conj() @ Ftr[m].T) ** 2; n = m.sum()
        Dc = 1.0 - Fk.mean(axis=1); Dc[m] = 1.0 - (Fk[m].sum(axis=1) - 1.0) / (n - 1); D[:, c] = Dc
    return D.min(axis=1)


VAL_DIR = os.path.join(HERE, 'validation')


def validate_job(job, data_dir, prefix, max_epochs):
    """job 0-11 = (split, pseudo-OOD fold), training data only. Saves summary metrics and the PER-SAMPLE
    validation scores (raw, MAD, labels) used by select_threshold.py to choose the test thresholds."""
    v = VARIANTS[0]; s, fold = job // N_FOLDS, job % N_FOLDS
    data = load_split(data_dir, prefix, s); X, y = data['Xtr'], data['ytr']
    rng = np.random.default_rng(1000 + 10 * s + fold)
    idx_in = rng.permutation(np.where(y != fold)[0]); n_val = int(0.2 * len(idx_in))
    val_in, tr_in, val_out = idx_in[:n_val], idx_in[n_val:], np.where(y == fold)[0]
    keep = sorted(set(np.unique(y[tr_in]).tolist())); y_tr = np.array([keep.index(c) for c in y[tr_in]])
    X_val = np.vstack([X[val_in], X[val_out]]); ood = np.r_[np.zeros(len(val_in), int), np.ones(len(val_out), int)]
    print(f'VALIDATION job {job}: split {s} ({data["family"]} = real OOD, untouched), pseudo-OOD fold {fold}', flush=True)
    Ftr, (Fval,), info = train_embed(X[tr_in], y_tr, [X_val], v, SEL_SEED, max_epochs, tag='[val] ')
    raw, mad = detect(Ftr, y_tr, Fval, v); ev = evaluate(raw, mad, ood)
    res = {'split': s, 'fold': fold, 'best_epoch': info['best_epoch'],
           'model': {'ROC_raw': ev['threshold_free']['ROC_AUC'], 'ROC_mad': ev['ROC_AUC_mad'],
                     'CADE_F1@3.5': [e for e in ev['sweep'] if e['T_MAD'] == 3.5][0]['F1']}}
    os.makedirs(VAL_DIR, exist_ok=True)
    json.dump(res, open(os.path.join(VAL_DIR, f's{s}_f{fold}.json'), 'w'), indent=2)
    np.savez_compressed(os.path.join(VAL_DIR, f'scores_s{s}_f{fold}.npz'), raw=raw, mad=mad, ood=ood)
    print(f'  VALIDATION {res["model"]}', flush=True)


# ----------------------------------------------------------------------------- selection (validation only)
def probe(data_dir, prefix, n=400):
    """Pre-check at initialization (no training): same- vs different-family fidelity vs the floor 1/256."""
    import torch
    v = VARIANTS[0]; circuit = make_circuit(n_layers(v))
    for s_ in range(N_SPLITS):
        data = load_split(data_dir, prefix, s_)
        pca = fit_pca(data['Xtr']); Z = pca.transform(data['Xtr']); mu, sc = Z.mean(0), float(Z[:, 0].std()) + 1e-8
        idx = np.random.default_rng(0).choice(len(Z), min(n, len(Z)), replace=False)
        A = torch.from_numpy(np.arctan((Z[idx] - mu) / sc).astype(np.float32)); y = data['ytr'][idx]
        set_seed(42); w = torch.randn(n_layers(v), N_QUBITS, 2) * INIT_STD
        with torch.no_grad():
            st = circuit(A, w, torch.ones(N_UPLOADS, N_QUBITS), torch.zeros(N_UPLOADS, N_QUBITS)).numpy()
        F = np.abs(st.conj() @ st.T) ** 2
        same = (y[:, None] == y[None, :]) & ~np.eye(len(y), dtype=bool)
        print(f'  split{s_} ({data["family"]:<12}) F_same {F[same].mean():.4f}  F_diff {F[y[:, None] != y[None, :]].mean():.4f}  '
              f'floor {1 / 2 ** N_QUBITS:.4f}')
    print('Healthy: both well above the floor and F_same > F_diff.')


def select_job(job, data_dir, prefix, max_epochs):
    """job -> (variant, split, fold). Uses ONLY the split's TRAINING data: known family `fold` becomes the
    pseudo-OOD family; the other two known families are split 80/20 into fold-train / validation-ID.
    The real held-out family and the test set are never touched."""
    nv = len(VARIANTS)
    vi, rest = job // (N_SPLITS * N_FOLDS), job % (N_SPLITS * N_FOLDS)
    s, fold = rest // N_FOLDS, rest % N_FOLDS
    v = VARIANTS[vi]
    data = load_split(data_dir, prefix, s)
    X, y = data['Xtr'], data['ytr']
    rng = np.random.default_rng(1000 + 10 * s + fold)
    idx_in = rng.permutation(np.where(y != fold)[0]); n_val = int(0.2 * len(idx_in))
    val_in, tr_in, val_out = idx_in[:n_val], idx_in[n_val:], np.where(y == fold)[0]
    keep = sorted(set(np.unique(y[tr_in]).tolist()))
    y_tr = np.array([keep.index(c) for c in y[tr_in]])
    X_val = np.vstack([X[val_in], X[val_out]])
    ood = np.r_[np.zeros(len(val_in), int), np.ones(len(val_out), int)]
    print(f'SELECTION job {job}: variant {vi} ({vname(v)}), split {s} ({data["family"]} = real OOD, untouched), '
          f'pseudo-OOD = known label {data["known_labels"][fold]}', flush=True)
    Ftr, (Fval,), info = train_embed(X[tr_in], y_tr, [X_val], v, SEL_SEED, max_epochs, tag='[sel] ')
    raw, mad = detect(Ftr, y_tr, Fval, v)
    ev = evaluate(raw, mad, ood)
    res = {'variant_index': vi, 'variant': v, 'split': s, 'fold': fold,
           'ROC_raw': ev['threshold_free']['ROC_AUC'], 'ROC_mad': ev['ROC_AUC_mad'],
           'CADE_F1@3.5': [e for e in ev['sweep'] if e['T_MAD'] == 3.5][0]['F1'],
           'best_epoch': info['best_epoch'], 'n_params': info['n_params']}
    os.makedirs(SEL_DIR, exist_ok=True)
    json.dump(res, open(os.path.join(SEL_DIR, f'v{vi}_s{s}_f{fold}.json'), 'w'), indent=2)
    print(f'  VALIDATION  ROC_raw {res["ROC_raw"]:.4f}  ROC_mad {res["ROC_mad"]:.4f}  '
          f'CADE_F1@3.5 {res["CADE_F1@3.5"]:.4f}', flush=True)


def select_summary():
    """Pre-declared criterion: mean over (split, fold) of (ROC_raw + ROC_mad) / 2 on VALIDATION data."""
    import glob
    rows = [json.load(open(f)) for f in glob.glob(os.path.join(SEL_DIR, 'v*_s*_f*.json'))]
    lines = []
    def log(s=''):
        print(s, flush=True); lines.append(s)
    log('=' * 100); log('QUANTUM v2 -- VARIANT SELECTION on validation data (training splits only; test never used)')
    log('Criterion (fixed in advance): mean over splits x folds of (ROC_raw + ROC_mad) / 2'); log('=' * 100)
    exp = N_SPLITS * N_FOLDS; best, table = None, []
    log(f"  {'#':>2}  {'variant':<48}{'runs':>6}{'ROC_raw':>10}{'ROC_mad':>10}{'CADE_F1@3.5':>13}{'criterion':>11}")
    for vi, v in enumerate(VARIANTS):
        rs = [r for r in rows if r['variant_index'] == vi]
        if not rs:
            log(f'  {vi:>2}  {vname(v):<48}{0:>6}   (not run)'); continue
        m = {k: float(np.mean([r[k] for r in rs])) for k in ['ROC_raw', 'ROC_mad', 'CADE_F1@3.5']}
        crit = (m['ROC_raw'] + m['ROC_mad']) / 2
        table.append({'variant_index': vi, 'variant': v, 'runs': len(rs), **m, 'criterion': crit})
        flag = '' if len(rs) == exp else f'  (incomplete: {len(rs)}/{exp})'
        log(f'  {vi:>2}  {vname(v):<48}{len(rs):>6}{m["ROC_raw"]:>10.4f}{m["ROC_mad"]:>10.4f}'
            f'{m["CADE_F1@3.5"]:>13.4f}{crit:>11.4f}{flag}')
        if len(rs) == exp and (best is None or crit > best['criterion']):
            best = table[-1]
    if best is None:
        log('\nNo variant has all selection runs yet.'); return
    log(f'\nSELECTED: variant {best["variant_index"]} -- {vname(best["variant"])}  (criterion {best["criterion"]:.4f})')
    json.dump({'selected': best, 'all': table}, open(os.path.join(SEL_DIR, 'selected_variant.json'), 'w'), indent=2)
    open(os.path.join(SEL_DIR, 'selection_summary.txt'), 'w').write('\n'.join(lines) + '\n')
    print(f'Saved: {SEL_DIR}/selected_variant.json, selection_summary.txt')


# ----------------------------------------------------------------------------- final runs
def load_variant(forced):
    return 0, VARIANTS[0]                         # fixed: Quantum v2's selected design


def run(task_id, data_dir, prefix, results_dir, max_epochs, forced_variant=None):
    import torch
    vi, v = load_variant(forced_variant)
    run_id, split, seed = task_info(task_id)
    data = load_split(data_dir, prefix, split)
    name = 'Quantum angle-8 (Quantum v2 design, 8 qubits, 8 layers, 4 uploads, 1-F, 192 params)'
    print(f'{name} | task {task_id} run {run_id} split {split} seed {seed} OOD={data["family"]}', flush=True)
    Ftr, (Fte,), info = train_embed(data['Xtr'], data['ytr'], [data['Xte']], v, seed, max_epochs)
    raw, mad = detect(Ftr, data['ytr'], Fte, v)
    # latent plot: kernel PCA (fidelity kernel) for fidelity variants, PCA of Bloch features for local
    from sklearn.decomposition import KernelPCA
    sub = np.random.default_rng(0).choice(len(Ftr), min(800, len(Ftr)), replace=False)
    if v['distance'] == 'local':
        p2 = PCA(n_components=2).fit(Ftr); lat_tr, lat_te, kind = p2.transform(Ftr[sub]), p2.transform(Fte), '2-D PCA of per-qubit Bloch vectors'
    else:
        K = np.abs(Ftr[sub].conj() @ Ftr[sub].T) ** 2; kp = KernelPCA(n_components=2, kernel='precomputed').fit(K)
        lat_tr, lat_te, kind = kp.transform(K), kp.transform(np.abs(Fte.conj() @ Ftr[sub].T) ** 2), 'kernel PCA (fidelity kernel)'
    save_dir = os.path.join(results_dir, f'run{run_id}', f'split{split}')
    os.makedirs(save_dir, exist_ok=True)
    meta = {'run_id': run_id, 'split_id': split, 'seed': seed, 'input': f'PCA-{PCA_DIM} -> shared scale -> arctan',
            'variant_index': vi, 'variant': v, 'architecture': f'{N_QUBITS} qubits, {n_layers(v)} layers, {N_UPLOADS} uploads',
            'n_trainable_params': info['n_params'], 'lr': LR, 'epochs_run': info['epochs_run'],
            'best_epoch': info['best_epoch'], 'best_loss': info['best_loss'], 'patience': PATIENCE,
            'train_time_sec': info['train_time_sec'], 'pca_explained_variance': float(info['pca'].explained_variance_ratio_.sum())}
    torch.save({'weights': info['weights'], 'scale_a': info['sa'], 'scale_b': info['sb'], 'variant': v,
                'pca_mean': info['mu'], 'shared_scale': info['scale']}, os.path.join(save_dir, 'model_weights.pt'))
    np.savez_compressed(os.path.join(save_dir, 'features.npz'), features_train=Ftr, features_test=Fte,
                        y_train=data['ytr'], y_test=data['yte'])
    np.save(os.path.join(save_dir, 'train_anomaly_scores.npy'), train_anomaly(Ftr, data['ytr'], v))
    np.save(os.path.join(save_dir, 'train_raw_scores.npy'), train_raw(Ftr, data['ytr'], v))
    s = save_task(save_dir, name, meta, raw, mad, data, info['hist'], None, None, None)
    dplot = dict(data); dplot['ytr'] = data['ytr'][sub]
    plot_task(os.path.join(save_dir, 'plots'), name, dplot, raw, mad, s['evaluation'], info['hist'],
              info['best_epoch'], lat_tr, lat_te, kind)
    e = s['evaluation']
    print(f'  DONE  ROC_raw {e["threshold_free"]["ROC_AUC"]:.4f}  ROC_mad {e["ROC_AUC_mad"]:.4f}  '
          f'TF_F1 {e["threshold_free"]["F1"]:.4f}  CADE_F1@3.5 {[x for x in e["sweep"] if x["T_MAD"] == 3.5][0]["F1"]:.4f}  '
          f'-> {save_dir}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Quantum v2: variant selection on validation data, then final runs')
    p.add_argument('cmd', help="a task 0-11, 'aggregate', 'probe', or 'validate' (with --job 0-11)")
    p.add_argument('--job', type=int, help=f'selection job 0-{len(VARIANTS) * N_SPLITS * N_FOLDS - 1}')
    p.add_argument('--variant', type=int, default=None, help='force a variant index (skip selection)')
    p.add_argument('--data-dir', default=DEFAULT_DATA)
    p.add_argument('--prefix', default=DEFAULT_PREFIX)
    p.add_argument('--results-dir', default='results')
    p.add_argument('--max-epochs', type=int, default=MAX_EPOCHS)
    a = p.parse_args()
    if a.cmd == 'probe':
        probe(a.data_dir, a.prefix)
    elif a.cmd == 'validate':
        validate_job(a.job, a.data_dir, a.prefix, a.max_epochs)
    elif a.cmd == 'aggregate':
        aggregate(os.path.abspath(os.path.expanduser(a.results_dir)), 'Quantum angle-8 (Quantum v2 design, 8 qubits, 192 params)')
    else:
        run(int(a.cmd), a.data_dir, a.prefix, os.path.abspath(os.path.expanduser(a.results_dir)), a.max_epochs, a.variant)
