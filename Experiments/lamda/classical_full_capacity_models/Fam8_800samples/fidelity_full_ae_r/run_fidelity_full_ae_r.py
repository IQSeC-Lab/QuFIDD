#!/usr/bin/env python3
"""
run_fidelity_full_ae_r.py -- classical_full_capacity_models/Fam8_800samples/fidelity_full_ae_r   (standalone)
Fidelity autoencoder CAPACITY-MATCHED TO CADE ORIGINAL (real: head 32 -> 8, a unit vector in R^8 (fidelity = squared cosine)): CADE's encoder trunk F-512-128-32 -> head,
unit-vector state, PLUS CADE's mirrored decoder (head -> 32 -> 128 -> 512 -> F) and CADE's loss weighting
(0.1 * metric loss + MSE). ~1.4M params, as CADE original. Same network, training and checkpointing as CADE;
only the latent geometry differs (fidelity + pure-state prototypes vs Euclidean distance + centroids).
Experiment: 8-family LAMDA, month-stratified, max 800 samples/family (cade_lamda_800_vt), FULL variance-thresholded
features (no PCA). 24 tasks = 8 held-out families x 3 seeds. Evaluation identical to all other scripts.
Usage (from classical_full_capacity_models/Fam8_800samples/fidelity_full_ae_r/):
  mkdir -p logs && sbatch run_fidelity_full_ae_r.slurm     # 24 tasks
  python3 run_fidelity_full_ae_r.py aggregate
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
PCA_DIM        = 8
THRESHOLDS     = [0.0, 2.0, 2.5, 3.0, 3.5]
MAD_T          = 3.5
MAD_B          = 1.4826
BATCH_SIZE     = 64
SIMILAR_RATIO  = 0.25
MAX_EPOCHS     = 250
PATIENCE       = 20
REL_MIN_DELTA  = 1e-4
LR             = 0.001
DEFAULT_DATA   = '~/projects/OOD/Data/cade_lamda_800_vt'
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
            'reconstruction': reconstruction, 'margin': margin, 'n_trainable_params': int(n_params), 'lr': LR,
            'epochs_run': len(hist), 'best_epoch': stop.best_epoch, 'best_loss': stop.best,
            'patience': PATIENCE, 'train_time_sec': round(time.time() - t0, 1)}
    os.makedirs(save_dir, exist_ok=True)
    torch.save(modules.state_dict(), os.path.join(save_dir, 'model_weights.pt'))
    _tr = centroid_detect(Ltr, data['ytr'], Ltr)
    np.save(os.path.join(save_dir, 'train_anomaly_scores.npy'), _tr[1])
    np.save(os.path.join(save_dir, 'train_raw_scores.npy'), _tr[0])
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






# ============================================================================= FULL-CAPACITY ADDITIONS
# CADE as released (Yang et al., USENIX Security 2021; github.com/whyisyoung/CADE), as in Group16:
#   no PCA (full variance-thresholded features), encoder F -> 512 -> 128 -> 32 -> N_known (ReLU hidden, LINEAR
#   bottleneck), Glorot-uniform init, Adam LR 1e-4, batch 64, similar ratio 0.25, 250 epochs, NO early stopping:
#   every 10 epochs the mean loss of those 10 epochs is compared with the best so far and the weights are saved if it
#   improved; the best checkpoint is scored. Detection: class mean centroids + per-class MAD (b = 1.4826), T = 3.5.
CADE_HIDDEN = [512, 128, 32]
CADE_LR     = 0.0001
CKPT_WINDOW = 10


class CadeCheckpoint:
    """CADE's checkpoint rule: every CKPT_WINDOW epochs, keep the weights if the window's mean loss improved."""
    def __init__(self):
        self.best, self.best_epoch, self.state = np.inf, 0, None

    def step(self, epoch, hist, state_fn):
        if epoch % CKPT_WINDOW == 0:
            w = float(np.mean(hist[-CKPT_WINDOW:]))
            if w < self.best:
                self.best, self.best_epoch = w, epoch
                self.state = copy.deepcopy(state_fn())
                return True
        return False


def glorot(i, o):
    import torch.nn as nn
    l = nn.Linear(i, o); nn.init.xavier_uniform_(l.weight); nn.init.zeros_(l.bias); return l


def mlp(dims):
    """ReLU MLP with a LINEAR last layer (CADE's encoder / decoder shape)."""
    import torch.nn as nn
    layers = []
    for i in range(len(dims) - 2):
        layers += [glorot(dims[i], dims[i + 1]), nn.ReLU()]
    layers += [glorot(dims[-2], dims[-1])]
    return nn.Sequential(*layers)


import argparse
def pure_prototypes(S, y, C):
    """Dominant eigenvector of each class's density matrix (a pure state, phase-invariant)."""
    import torch
    St = torch.as_tensor(S); V = []
    for c in range(C):
        P = St[torch.as_tensor(y) == c]; rho = P.T @ P.conj() / len(P)
        V.append(torch.linalg.eigh(rho)[1][:, -1].conj())
    return torch.stack(V)


def prod_fid(Sa, Sb):
    """Pairwise product fidelity between two lists of register states -> [Na, Nb] (numpy)."""
    F = 1.0
    for A, B in zip(Sa, Sb):
        F = F * np.abs(A.conj() @ B.T) ** 2
    return F


def proto_detect(Str, ytr, Sev):
    """Nearest-prototype distance d_c = 1 - prod_r |<v_rc|psi_r>|^2, + CADE's per-class MAD score."""
    C = len(np.unique(ytr)); Vs = [pure_prototypes(S, ytr, C).numpy() for S in Str]
    Ptr, Pev = 1.0, 1.0
    for S, Se, V in zip(Str, Sev, Vs):
        Ptr = Ptr * np.abs(S @ V.T) ** 2; Pev = Pev * np.abs(Se @ V.T) ** 2
    Dtr, Dev = 1 - Ptr, 1 - Pev; A = []
    for c in range(C):
        med, mad = _mad_stats(Dtr[ytr == c, c]); A.append(np.abs(Dev[:, c] - med) / mad)
    return Dev.min(1), np.min(A, axis=0)




# ============================================================================= MODEL-SPECIFIC CODE -- fidelity autoencoder (full features, CADE-matched)
# The fidelity geometry of hybrid v7 / its classical twins, at CADE's full capacity:
#   psi = normalize( MLP(x) ),  MLP = CADE's encoder trunk F -> 512 -> 128 -> 32 (ReLU) -> head (linear)
#   COMPLEX: head 32 -> 16, read as a unit vector in C^8 (same state space as v7's 3-qubit state and the complex twin)
#   REAL:    head 32 -> 8,  a unit vector in R^8 (fidelity = squared cosine)
# Same fidelity |<a|b>|^2, pure-state prototypes (dominant eigenvector of each family's density matrix), losses
# (pair contrastive on 1 - F with margin 1 + prototype cross-entropy, tau 0.1) and scoring (nearest prototype + MAD)
# as v7 and the twins. Trained with CADE's protocol (LR 1e-4, batch 64, 250 epochs, 10-epoch checkpointing).
# CAPACITY-MATCHED TO CADE ORIGINAL: like CADE, it has a mirrored decoder (head output -> 32 -> 128 -> 512 -> F, linear
# output) and CADE's loss weighting: loss = 0.1 * (pair + prototype) + MSE reconstruction. So the network, training,
# parameter count (~1.4M) and loss structure all match CADE original; the only difference is the latent geometry
# (unit vectors + fidelity + prototypes instead of Euclidean distance + centroids).
STATE_DIM = 8
MARGIN, TAU, LAMBDA_P = 1.0, 0.1, 1.0
COMPLEX = False
NAME = 'Fidelity autoencoder, full features, real, CADE-matched (F-512-128-32-8 -> R^8 + mirrored decoder)'


def to_state(g):
    """Head output -> unit vector (C^8 from 16 outputs, or R^8 stored as complex)."""
    import torch
    z = torch.complex(g[:, :STATE_DIM], g[:, STATE_DIM:]) if COMPLEX else g.to(torch.complex64)
    return z / torch.linalg.vector_norm(z, dim=1, keepdim=True).clamp(min=1e-8)


def embed_fn(trunk):
    return lambda X: to_state(trunk(X))


def run(task_id, data_dir, prefix, results_dir, max_epochs):
    import torch, torch.optim as optim
    from sklearn.decomposition import KernelPCA
    run_id, split, seed = task_info(task_id)
    data = load_split(data_dir, prefix, split)
    Xtr, Xte, ytr = data['Xtr'], data['Xte'], data['ytr']
    F, C = Xtr.shape[1], len(np.unique(ytr))
    set_seed(seed)
    dims = [F] + CADE_HIDDEN + [2 * STATE_DIM if COMPLEX else STATE_DIM]
    trunk = mlp(dims); st = embed_fn(trunk); decoder = mlp(dims[::-1])
    modules = torch.nn.ModuleList([trunk, decoder])
    n_params = sum(p.numel() for p in modules.parameters())
    opt = optim.Adam(modules.parameters(), lr=CADE_LR, eps=1e-7)
    print(f'{NAME} | task {task_id} run {run_id} split {split} seed {seed} OOD={data["family"]}', flush=True)
    print(f'  features {F}, MLP {"-".join(map(str, dims))} -> unit vector in {"C" if COMPLEX else "R"}^{STATE_DIM}, '
          f'+ mirrored decoder, params {n_params}, LR {CADE_LR}, {max_epochs} epochs, checkpoint every {CKPT_WINDOW}', flush=True)

    def states(X):
        with torch.no_grad():
            return torch.cat([st(torch.from_numpy(X[i:i + 1024])) for i in range(0, len(X), 1024)])

    ck, hist, t0 = CadeCheckpoint(), [], time.time()
    for epoch in range(1, max_epochs + 1):
        V = pure_prototypes(states(Xtr), ytr, C).detach()          # prototypes refreshed every epoch
        bc, bx, by = epoch_batches(Xtr, ytr); ls = []
        for b in range(bc):
            xb, yb = torch.from_numpy(bx[b]), torch.from_numpy(by[b])
            opt.zero_grad()
            g = trunk(xb); S = to_state(g); h = S.shape[0] // 2
            ov = (S[:h].conj() * S[h:]).sum(-1); d = 1.0 - (ov.real ** 2 + ov.imag ** 2)
            same = (yb[:h] == yb[h:]).float()
            pair = (same * d + (1 - same) * torch.relu(MARGIN - d)).mean()
            proto = torch.nn.functional.cross_entropy((S @ V.T).abs() ** 2 / TAU, yb)
            loss = 0.1 * (pair + LAMBDA_P * proto) + ((decoder(g) - xb) ** 2).mean(1).mean()
            loss.backward(); opt.step(); ls.append(loss.item())
        hist.append(float(np.mean(ls)))
        saved = ck.step(epoch, hist, modules.state_dict)
        print(f'  epoch {epoch:3d} loss {hist[-1]:.5f}' + (f'  [checkpoint: window mean {ck.best:.5f}]' if saved else ''),
              flush=True)
    if ck.state is not None:
        modules.load_state_dict(ck.state)
    Ftr, Fte = [states(Xtr).numpy()], [states(Xte).numpy()]
    raw, mad = proto_detect(Ftr, ytr, Fte)
    sub = np.random.default_rng(0).choice(len(Ftr[0]), min(800, len(Ftr[0])), replace=False)
    Fsub = [S[sub] for S in Ftr]
    kp = KernelPCA(n_components=2, kernel='precomputed').fit(prod_fid(Fsub, Fsub))
    save_dir = os.path.join(results_dir, f'run{run_id}', f'split{split}'); os.makedirs(save_dir, exist_ok=True)
    meta = {'run_id': run_id, 'split_id': split, 'seed': seed, 'input': f'full features ({F})',
            'architecture': f'MLP {"-".join(map(str, dims))} -> unit vector in {"C" if COMPLEX else "R"}^{STATE_DIM}, '
                            f'fidelity, pure-state prototypes; mirrored decoder (linear output)', 'loss': '0.1*(pair + prototype) + MSE',
            'n_trainable_params': int(n_params), 'lr': CADE_LR, 'epochs_run': len(hist),
            'best_epoch': ck.best_epoch, 'best_loss': ck.best, 'checkpoint_window': CKPT_WINDOW,
            'train_time_sec': round(time.time() - t0, 1)}
    torch.save(modules.state_dict(), os.path.join(save_dir, 'model_weights.pt'))
    np.savez_compressed(os.path.join(save_dir, 'states.npz'), y_train=ytr, y_test=data['yte'],
                        states_train_r0=Ftr[0], states_test_r0=Fte[0])
    tr_raw, tr_A = proto_detect(Ftr, ytr, Ftr)
    np.save(os.path.join(save_dir, 'train_anomaly_scores.npy'), tr_A)
    np.save(os.path.join(save_dir, 'train_raw_scores.npy'), tr_raw)
    s = save_task(save_dir, NAME, meta, raw, mad, data, hist,
                  kp.transform(prod_fid(Ftr, Fsub)), kp.transform(prod_fid(Fte, Fsub)), 'kernel PCA (fidelity kernel)')
    e = s['evaluation']
    print(f'  DONE  ROC_raw {e["threshold_free"]["ROC_AUC"]:.4f}  ROC_mad {e["ROC_AUC_mad"]:.4f}  '
          f'CADE_F1@3.5 {[x for x in e["sweep"] if x["T_MAD"] == 3.5][0]["F1"]:.4f}  -> {save_dir}', flush=True)


if __name__ == '__main__':
    standard_main(f'Full-capacity: {NAME}', run, NAME)
