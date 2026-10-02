#!/usr/bin/env python3
"""
run_quantum_amp_256.py -- pca256_experiment / quantum_amp_256   (standalone)
==========================================================================
Quantum: amplitude encoding of all 256 PCA values + angle RE-UPLOADING, on 8 qubits.
  input      PCA-255 + one constant slot (= mean training norm), L2-normalized -> 256 amplitudes
             (the constant slot keeps the length of the PCA vector after normalization)
  encoding   |0> -> |x> (amplitude encoding of all 256 values, via the reflection V(x) = I - 2vv^T/|v|^2)
  circuit    22 trainable layers (RY, RZ on every qubit + CNOT ring). Before layers 8 and 15 the data is
             RE-UPLOADED as angles: RY(a_k * phi_k + b_k) on qubit k, phi = arctan-scaled top-8 principal
             components, a/b trainable (as Quantum v2). Re-uploading makes fidelities depend on the weights
             (single amplitude encoding is unitarily invariant) and the embedding nonlinear.
             Why angle re-uploads and not repeated reflections: in 256 dimensions V(x) is a rank-1 change
             of the identity, so re-applied reflections barely alter a generic state; measured on test data,
             the fidelities then hardly depend on the weights (dependence 0.005 vs 0.012 with angle re-uploads).
  params     22 x 8 x 2 = 352 circuit + 2 x 8 x 2 = 32 input scaling = 384
             (classical encoder: 18,627 -> input-dimension matched, not capacity-matched)
  distance   1 - fidelity (as the selected Quantum v2), margin 1
  scoring    density-matrix prototypes, d_c = 1 - <psi|rho_c|psi>, + CADE's MAD
  simulator  exact PyTorch statevector simulation (checked against PennyLane to 1e-15)
Experiment: 4-class LAMDA, INPUT-DIMENSION-MATCHED (not capacity-matched). Every model reads
PCA-256 fit on the split's training set. No variant selection: one fixed configuration per model.
Default run = pilot (seed 42, the 4 families = tasks 0-3); tasks 4-11 add seeds 123 and 456.
Shared with all scripts: CADE pairing (batch 64), Adam LR 0.001, early stopping (patience 20,
rel 1e-4, max 250, best weights), MAD detection, metrics, threshold sweep, plots.
Usage (from pca256_experiment/quantum_amp_256/):
  python3 run_quantum_amp_256.py probe             # pre-check on real data, no training (1-2 min)
  mkdir -p logs && sbatch run_quantum_amp_256.slurm  (pilot: tasks 0-3)
  python3 run_quantum_amp_256.py aggregate
"""

# =============================================================================
# SHARED CODE -- identical in all pca256_experiment scripts
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
N_SPLITS       = 4
N_TASKS        = 3 * N_SPLITS
PCA_DIM        = 256             # PCA-256 input (quantum: PCA-255 + constant slot)
THRESHOLDS     = [0.0, 2.0, 2.5, 3.0, 3.5]
MAD_T          = 3.5
MAD_B          = 1.4826
BATCH_SIZE     = 64
SIMILAR_RATIO  = 0.25
MAX_EPOCHS     = 250
PATIENCE       = 20
REL_MIN_DELTA  = 1e-4
LR             = 0.001
DEFAULT_DATA   = '~/projects/OOD/Data/cade_lamda_4class_vt'
DEFAULT_PREFIX = 'drebin_new'
FLAG_ALL_F1    = 0.769          # 1000 OOD vs 600 known on the 4-class LAMDA test sets


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
                row = {'run': r, 'split': s, 'family': d['ood_family'], 'n_params': d.get('n_trainable_params', 0),
                       'best_epoch': d.get('best_epoch', 0)}
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
    log(f'  Threshold-free F1 flag-all floor = {FLAG_ALL_F1}. std = population std.')
    log(f'  Params: {rows[0]["n_params"]}   best epochs: {[x["best_epoch"] for x in rows]}')
    with open(os.path.join(results_dir, 'aggregate_summary.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    with open(os.path.join(results_dir, 'aggregate.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['run', 'split', 'family', 'n_params', 'best_epoch'] + keys)
        w.writeheader(); w.writerows(rows)
    # plots
    pdir = os.path.join(results_dir, 'aggregate_plots'); os.makedirs(pdir, exist_ok=True)
    fig, a = plt.subplots(figsize=(6, 4))
    m = [np.mean([x[f'F1@T{T}'] for x in rows]) for T in THRESHOLDS]; sd = [np.std([x[f'F1@T{T}'] for x in rows]) for T in THRESHOLDS]
    a.errorbar(THRESHOLDS, m, yerr=sd, marker='o', capsize=3)
    a.axhline(FLAG_ALL_F1, color='k', ls=':', lw=1, label='flag-all F1 (T=0 floor)')
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
def run_cade_mini(task_id, data_dir, prefix, results_dir, hidden, reconstruction, model_name, max_epochs=MAX_EPOCHS):
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
            con = (same * dist + (1 - same) * torch.relu(10.0 - dist)).mean()
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
# MODEL-SPECIFIC CODE -- quantum, amplitude re-uploading, PCA-256 on 8 qubits
# =============================================================================
import argparse

N_QUBITS   = 8
DIM        = 2 ** N_QUBITS          # 256 amplitudes
N_PCA      = DIM - 1                # PCA-255 + 1 constant slot (keeps the vector's length)
N_REUP     = 2                      # angle re-uploads of the top-8 principal components (after the amplitude encoding)
N_LAYERS   = 22                     # trainable layers: 22 x 8 x 2 = 352, + input scaling 2 x 8 x 2 = 32 -> 384
UPLOAD_AT  = (8, 15)                # re-uploads before layers 8 and 15 (7-8 layers between uploads)
INIT_STD   = 0.1
MARGIN     = 1.0                    # both distances lie in [0, 1]
DISTANCE   = 'fid_lin'              # 1 - fidelity (as the selected Quantum v2); 'local' = per-qubit Bloch distance
NAME       = 'Quantum amplitude+angle re-upload (8 qubits, PCA-256, 22 layers, 1-F, 384 params)'


# ----------------------------------------------------------------------------- statevector simulator (PyTorch)
def _cnot_ring_perm(n):
    """Basis-state permutation of the CNOT ring 0->1->...->(n-1)->0 (wire 0 = most significant bit).
    CNOTs are permutation matrices, so the whole ring is one gather."""
    perm = np.arange(2 ** n)
    for c, t in [(k, k + 1) for k in range(n - 1)] + [(n - 1, 0)]:
        bits = (perm[:, None] >> (n - 1 - np.arange(n))) & 1          # bits of the CURRENT source index
        new = perm.copy()
        flip = bits[:, c] == 1
        new[flip] = perm[flip] ^ (1 << (n - 1 - t))
        perm = new
    # perm[i] = index the amplitude at i moves to; gather needs the inverse
    inv = np.empty_like(perm); inv[perm] = np.arange(2 ** n)
    return inv


class Sim:
    def __init__(self, n):
        import torch
        self.n = n
        self.inv = torch.from_numpy(_cnot_ring_perm(n)).long()

    def rot_layer(self, psi, th):
        """RY(th[k,0]) then RZ(th[k,1]) on every qubit k. psi: [B, 2^n] complex."""
        import torch
        B = psi.shape[0]; t = psi.reshape([B] + [2] * self.n)
        for k in range(self.n):
            c, s = torch.cos(th[k, 0] / 2), torch.sin(th[k, 0] / 2)
            ez = torch.exp(-0.5j * th[k, 1])
            U = torch.stack([torch.stack([c * ez, -s * ez]), torch.stack([s * ez.conj(), c * ez.conj()])])  # RZ @ RY
            t = torch.movedim(torch.tensordot(t, U.T, dims=([k + 1], [0])), -1, k + 1)
        return t.reshape(B, -1)

    def cnot_ring(self, psi):
        return psi[:, self.inv]


def reflect(psi, xh):
    """Data re-upload: V(x) psi with V(x) = I - 2 v v^T / |v|^2, v = e0 - xh (real, unit xh).
    V(x)|0> = |xh>, so the first application is plain amplitude encoding."""
    import torch
    v = -xh.clone(); v[:, 0] = v[:, 0] + 1.0
    nv = (v * v).sum(1, keepdim=True).clamp(min=1e-12)
    vc = v.to(psi.dtype)
    return psi - 2.0 * vc * ((vc.conj() * psi).sum(1, keepdim=True) / nv)


def angle_upload(psi, ang, a, b, n):
    """Per-sample data re-upload: RY(a_k * ang_k + b_k) on every qubit k (trainable input scaling a, b)."""
    import torch
    B = psi.shape[0]; t = psi.reshape([B] + [2] * n); shape = [B] + [1] * (n - 1)
    th = ang * a + b                                                      # [B, n]
    for k in range(n):
        c = torch.cos(th[:, k] / 2).to(psi.dtype).view(shape); s_ = torch.sin(th[:, k] / 2).to(psi.dtype).view(shape)
        t0, t1 = t.select(k + 1, 0), t.select(k + 1, 1)
        t = torch.stack([c * t0 - s_ * t1, s_ * t0 + c * t1], dim=k + 1)
    return t.reshape(B, -1)


def bloch(psi, n):
    """Per-qubit Bloch vectors: [B, 2^n] -> [B, n*3]."""
    import torch
    B = psi.shape[0]; t = psi.reshape([B] + [2] * n); out = []
    for k in range(n):
        m = torch.movedim(t, k + 1, 1).reshape(B, 2, -1)
        rho = m @ m.conj().transpose(1, 2)
        out += [2 * rho[:, 0, 1].real, -2 * rho[:, 0, 1].imag, (rho[:, 0, 0] - rho[:, 1, 1]).real]
    return torch.stack(out, dim=1)


def forward(sim, xh, ang, weights, sa, sb):
    import torch
    B = xh.shape[0]
    e0 = torch.zeros(B, DIM, dtype=torch.complex64); e0[:, 0] = 1.0
    psi = reflect(e0, xh)                                        # amplitude encoding of all 256 values
    u = 0
    for l in range(N_LAYERS):
        if l in UPLOAD_AT:
            psi = angle_upload(psi, ang, sa[u], sb[u], N_QUBITS); u += 1     # angle re-upload (top-8 PCs)
        psi = sim.cnot_ring(sim.rot_layer(psi, weights[l]))
    return psi


# ----------------------------------------------------------------------------- input: PCA-255 + constant slot
class AmpInput:
    def fit(self, Xtr):
        self.pca = PCA(n_components=N_PCA, svd_solver='full').fit(Xtr)
        Z = self.pca.transform(Xtr)
        self.c = float(np.linalg.norm(Z, axis=1).mean())                        # constant slot = mean norm
        self.mu, self.scale = Z[:, :N_QUBITS].mean(0), float(Z[:, 0].std()) + 1e-8  # for the angle re-uploads
        return self

    def __call__(self, X):
        """Returns (amplitudes [n, 256], angles [n, 8])."""
        Z = self.pca.transform(X)
        V = np.hstack([Z, np.full((len(Z), 1), self.c)])
        amp = (V / np.linalg.norm(V, axis=1, keepdims=True)).astype(np.float32)
        ang = np.arctan((Z[:, :N_QUBITS] - self.mu) / self.scale).astype(np.float32)
        return amp, ang


def local_distance(fa, fb):
    import torch
    return torch.sqrt(((fa - fb) ** 2).sum(1) / (4.0 * N_QUBITS) + 1e-10)


def train_embed(Xtr, ytr, X_eval, seed, max_epochs):
    import torch, torch.optim as optim
    inp = AmpInput().fit(Xtr); amp_tr, ang_tr = inp(Xtr); Atr = np.hstack([amp_tr, ang_tr])
    sim = Sim(N_QUBITS)
    set_seed(seed)
    weights = torch.nn.Parameter(torch.randn(N_LAYERS, N_QUBITS, 2) * INIT_STD)
    sa = torch.nn.Parameter(torch.ones(N_REUP, N_QUBITS)); sb = torch.nn.Parameter(torch.zeros(N_REUP, N_QUBITS))
    params = [weights, sa, sb]; n_params = int(sum(p.numel() for p in params))
    opt = optim.Adam(params, lr=LR)
    def feat(A):                                                           # A = [amplitudes | angles]
        psi = forward(sim, A[:, :DIM], A[:, DIM:], weights, sa, sb)
        return psi if DISTANCE == 'fid_lin' else bloch(psi, N_QUBITS)
    print(f'  {NAME}: {n_params} trainable params', flush=True)
    stop, hist, t0 = EarlyStopper(), [], time.time()
    snap = lambda: {'w': weights.detach().clone(), 'a': sa.detach().clone(), 'b': sb.detach().clone()}
    stop.state = snap()
    for epoch in range(1, max_epochs + 1):
        bc, bx, by = epoch_batches(Atr, ytr)
        ls = []
        for b in range(bc):
            A, yb = torch.from_numpy(bx[b]), torch.from_numpy(by[b])
            opt.zero_grad()
            f = feat(A); h = f.shape[0] // 2
            if DISTANCE == 'fid_lin':
                ov = (f[:h].conj() * f[h:]).sum(-1); d = 1.0 - (ov.real ** 2 + ov.imag ** 2)
            else:
                d = local_distance(f[:h], f[h:])
            same = (yb[:h] == yb[h:]).float()
            loss = (same * d + (1 - same) * torch.relu(MARGIN - d)).mean()
            loss.backward(); opt.step(); ls.append(loss.item())
        hist.append(float(np.mean(ls)))
        done = stop.step(epoch, hist[-1], snap)
        print(f'  epoch {epoch:3d} loss {hist[-1]:.5f} best {stop.best:.5f} (ep {stop.best_epoch}) '
              f'no_improve {stop.since} ({time.time() - t0:.0f}s)', flush=True)
        if done:
            break
    with torch.no_grad():
        weights.copy_(stop.state['w']); sa.copy_(stop.state['a']); sb.copy_(stop.state['b'])
        F = lambda X: np.vstack([feat(torch.from_numpy(np.hstack(inp(X[i:i + 512])))).numpy() for i in range(0, len(X), 512)])
        Ftr, Fev = F(Xtr), [F(X) for X in X_eval]
    info = {'n_params': n_params, 'sa': sa.detach(), 'sb': sb.detach(), 'epochs_run': len(hist), 'best_epoch': stop.best_epoch,
            'best_loss': (None if not np.isfinite(stop.best) else stop.best), 'hist': hist,
            'train_time_sec': round(time.time() - t0, 1), 'inp': inp, 'weights': weights.detach()}
    return Ftr, Fev, info


def prototype_train_scores(Ftr, ytr):
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


def run(task_id, data_dir, prefix, results_dir, max_epochs):
    import torch
    run_id, split, seed = task_info(task_id)
    data = load_split(data_dir, prefix, split)
    print(f'{NAME} | task {task_id} run {run_id} split {split} seed {seed} OOD={data["family"]}', flush=True)
    Ftr, (Fte,), info = train_embed(data['Xtr'], data['ytr'], [data['Xte']], seed, max_epochs)
    if DISTANCE == 'fid_lin':
        raw, mad = prototype_detect(Ftr, data['ytr'], Fte)       # 1 - <psi|rho_c|psi>, density-matrix prototypes
        from sklearn.decomposition import KernelPCA
        sub_ = np.random.default_rng(0).choice(len(Ftr), min(800, len(Ftr)), replace=False)
        K = np.abs(Ftr[sub_].conj() @ Ftr[sub_].T) ** 2; kp = KernelPCA(n_components=2, kernel='precomputed').fit(K)
        lat_tr, lat_te, lat_kind = kp.transform(np.abs(Ftr.conj() @ Ftr[sub_].T) ** 2), kp.transform(np.abs(Fte.conj() @ Ftr[sub_].T) ** 2), 'kernel PCA (fidelity kernel)'
        trainA = prototype_train_scores(Ftr, data['ytr'])
    else:
        raw, mad = centroid_detect(Ftr, data['ytr'], Fte)        # Euclidean on per-qubit Bloch vectors
        p2 = PCA(n_components=2).fit(Ftr)
        lat_tr, lat_te, lat_kind = p2.transform(Ftr), p2.transform(Fte), '2-D PCA of the per-qubit Bloch vectors'
        trainA = centroid_detect(Ftr, data['ytr'], Ftr)[1]
    save_dir = os.path.join(results_dir, f'run{run_id}', f'split{split}'); os.makedirs(save_dir, exist_ok=True)
    meta = {'run_id': run_id, 'split_id': split, 'seed': seed, 'input': f'PCA-{N_PCA} + constant slot, amplitude-encoded',
            'pca_explained_variance': float(info['inp'].pca.explained_variance_ratio_.sum()),
            'architecture': f'{N_QUBITS} qubits, amplitude encoding + {N_REUP} angle re-uploads (top-{N_QUBITS} PCs, trainable scaling), {N_LAYERS} layers RY+RZ+CNOT ring, distance {DISTANCE}',
            'n_trainable_params': info['n_params'], 'lr': LR, 'epochs_run': info['epochs_run'],
            'best_epoch': info['best_epoch'], 'best_loss': info['best_loss'], 'patience': PATIENCE,
            'train_time_sec': info['train_time_sec']}
    torch.save({'weights': info['weights'], 'scale_a': info['sa'], 'scale_b': info['sb']}, os.path.join(save_dir, 'model_weights.pt'))
    np.savez_compressed(os.path.join(save_dir, 'features.npz'), features_train=Ftr, features_test=Fte,
                        y_train=data['ytr'], y_test=data['yte'])
    np.save(os.path.join(save_dir, 'train_anomaly_scores.npy'), trainA)
    s = save_task(save_dir, NAME, meta, raw, mad, data, info['hist'], lat_tr, lat_te, lat_kind)
    e = s['evaluation']
    print(f'  DONE  ROC_raw {e["threshold_free"]["ROC_AUC"]:.4f}  ROC_mad {e["ROC_AUC_mad"]:.4f}  '
          f'CADE_F1@3.5 {[x for x in e["sweep"] if x["T_MAD"] == 3.5][0]["F1"]:.4f}  -> {save_dir}', flush=True)


def probe(data_dir, prefix, n=400):
    """Pre-check on real data, no training: (1) features depend on the weights (no unitary invariance),
    (2) at initialization, same-family pairs are closer than different-family pairs (no concentration)."""
    import torch
    sim = Sim(N_QUBITS)
    for s in range(N_SPLITS):
        data = load_split(data_dir, prefix, s)
        idx = np.random.default_rng(0).choice(len(data['ytr']), min(n, len(data['ytr'])), replace=False)
        amp, ang = AmpInput().fit(data['Xtr'])(data['Xtr'][idx]); y = data['ytr'][idx]
        A, G = torch.from_numpy(amp), torch.from_numpy(ang)
        ones, zeros = torch.ones(N_REUP, N_QUBITS), torch.zeros(N_REUP, N_QUBITS)
        with torch.no_grad():
            set_seed(1); p1 = forward(sim, A, G, torch.randn(N_LAYERS, N_QUBITS, 2) * INIT_STD, ones, zeros)
            set_seed(2); p2 = forward(sim, A, G, torch.randn(N_LAYERS, N_QUBITS, 2) * INIT_STD, ones, zeros)
        same = (y[:, None] == y[None, :]) & ~np.eye(len(y), dtype=bool); diff = y[:, None] != y[None, :]
        F1 = (p1.conj() @ p1.T).abs().numpy() ** 2; F2 = (p2.conj() @ p2.T).abs().numpy() ** 2
        An = amp; C2 = (An @ An.T) ** 2
        print(f'  split{s} ({data["family"]:<12}) fidelity same {F1[same].mean():.3f} diff {F1[diff].mean():.3f} '
              f'| input cos^2 same {C2[same].mean():.3f} diff {C2[diff].mean():.3f} '
              f'| weight dependence mean|F(theta1)-F(theta2)| {np.abs(F1 - F2)[~np.eye(len(y), dtype=bool)].mean():.4f} '
              f'| random-state floor {1 / DIM:.4f}')
    print('Healthy: fidelities well above the floor, same > diff, and a non-zero weight dependence (no unitary invariance).')


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'probe':
        ap = argparse.ArgumentParser(); ap.add_argument('cmd'); ap.add_argument('--data-dir', default=DEFAULT_DATA)
        ap.add_argument('--prefix', default=DEFAULT_PREFIX); a = ap.parse_args(); probe(a.data_dir, a.prefix)
    else:
        standard_main('PCA-256 experiment: quantum amplitude re-uploading', run, NAME)
