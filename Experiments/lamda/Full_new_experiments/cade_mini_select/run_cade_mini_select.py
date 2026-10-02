#!/usr/bin/env python3
"""
run_cade_mini_select.py -- Full_new_experiments / cade_mini_select   (standalone script)
======================================================================================
CADE-mini with the SAME validation-based selection the quantum models got, so the final
comparison gives both sides an equal tuning procedure.
Candidates (fixed in advance), all ~200 parameters, on PCA-5:
  design  enc        encoder 5-22-3, contrastive only (201 params)
          full_cade  encoder 5-11-3 + decoder, 0.1*contrastive + MSE (CADE's weighting, 206 params)
          full_eq    encoder 5-11-3 + decoder, 1.0*contrastive + MSE (206 params)
  margin  5, 10 (CADE), 20
  lr      0.001, 0.003
  -> 18 variants x (4 splits x 3 pseudo-OOD folds) = 216 validation jobs (each ~0.5-1 min)
Same folds, same converged training rule (cap 250, patience 20), same criterion
(mean (ROC_raw + ROC_mad) / 2) as quantum_v2 / quantum_v3. Test data is never used for the choice.

Workflow (from Full_new_experiments/cade_mini_select/):
  python3 run_cade_mini_select.py variants
  mkdir -p logs selection && sbatch run_select.slurm        # 216 validation jobs
  python3 run_cade_mini_select.py select-summary
  sbatch run_cade_mini_select.slurm                          # final 12 tasks, selected variant
  python3 run_cade_mini_select.py aggregate
"""

# =============================================================================
# SHARED CODE -- identical in all four Full_new_experiments scripts
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
PCA_DIM        = 5
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
# MODEL-SPECIFIC CODE -- CADE-mini with validation-based selection
# =============================================================================
import argparse

HERE       = os.path.dirname(os.path.abspath(__file__))
SEL_DIR    = os.path.join(HERE, 'selection')
SEL_EPOCHS = 250
SEL_SEED   = 42
N_FOLDS    = 3

# Candidates (fixed in advance), all ~200 parameters, mirroring the quantum selection effort:
#   design  : encoder-only 5-22-3 (201 params, contrastive only) |
#             full 5-11-3 + decoder (206 params), reconstruction weight lam: loss = lam_c*con + MSE with
#             CADE's 0.1*con + MSE ('full_cade') or 1.0*con + MSE ('full_eq')
#   margin  : 5, 10 (CADE), 20
#   lr      : 0.001, 0.003
DESIGNS = {'enc': {'hidden': [22], 'decoder': False, 'con_w': 1.0},
           'full_cade': {'hidden': [11], 'decoder': True, 'con_w': 0.1},
           'full_eq': {'hidden': [11], 'decoder': True, 'con_w': 1.0}}
VARIANTS = [{'design': d, 'margin': m, 'lr': lr} for d in DESIGNS for m in (5.0, 10.0, 20.0) for lr in (0.001, 0.003)]


def vname(v):
    return f"{v['design']} / margin {v['margin']:g} / lr {v['lr']:g}"


def train_embed(Xtr, ytr, X_eval, seed, max_epochs, v, tag=''):
    import torch, torch.nn as nn, torch.optim as optim
    pca = fit_pca(Xtr)
    Ztr = pca.transform(Xtr).astype(np.float32)
    C = len(np.unique(ytr)); D_ = DESIGNS[v['design']]
    set_seed(seed)

    def dense(i, o):
        l = nn.Linear(i, o); nn.init.xavier_uniform_(l.weight); nn.init.zeros_(l.bias); return l
    dims = [PCA_DIM] + D_['hidden'] + [C]
    enc = []
    for i in range(len(dims) - 2):
        enc += [dense(dims[i], dims[i + 1]), nn.ReLU()]
    enc += [dense(dims[-2], dims[-1])]
    encoder = nn.Sequential(*enc); decoder = None
    if D_['decoder']:
        dec = []
        for i in range(len(dims) - 1, 1, -1):
            dec += [dense(dims[i], dims[i - 1]), nn.ReLU()]
        dec += [dense(dims[1], dims[0])]
        decoder = nn.Sequential(*dec)
    mods = nn.ModuleList([encoder] + ([decoder] if decoder is not None else []))
    n_params = sum(p.numel() for p in mods.parameters())
    opt = optim.Adam(mods.parameters(), lr=v['lr'], eps=1e-7)
    print(f'  {tag}variant: {vname(v)} | encoder {"-".join(map(str, dims))}, decoder {"yes" if decoder is not None else "no"}, '
          f'{n_params} params', flush=True)
    stop, hist, t0 = EarlyStopper(), [], time.time()
    stop.state = copy.deepcopy(mods.state_dict())
    for epoch in range(1, max_epochs + 1):
        bc, bx, by = epoch_batches(Ztr, ytr)
        ls = []
        for b in range(bc):
            xb, yb = torch.from_numpy(bx[b]), torch.from_numpy(by[b])
            opt.zero_grad()
            z = encoder(xb); h = z.shape[0] // 2
            dist = torch.sqrt(((z[:h] - z[h:]) ** 2).sum(1) + 1e-10)
            same = (yb[:h] == yb[h:]).float()
            con = (same * dist + (1 - same) * torch.relu(v['margin'] - dist)).mean()
            loss = D_['con_w'] * con + (((decoder(z) - xb) ** 2).mean(1).mean() if decoder is not None else 0.0)
            loss.backward(); opt.step(); ls.append(loss.item())
        hist.append(float(np.mean(ls)))
        done = stop.step(epoch, hist[-1], mods.state_dict)
        if epoch % 10 == 0 or done:
            print(f'  {tag}epoch {epoch:3d} loss {hist[-1]:.5f} best {stop.best:.5f} (ep {stop.best_epoch})', flush=True)
        if done:
            break
    mods.load_state_dict(stop.state)
    with torch.no_grad():
        f = lambda X: encoder(torch.from_numpy(pca.transform(X).astype(np.float32))).numpy()
        Ltr, Lev = f(Xtr), [f(X) for X in X_eval]
    info = {'n_params': int(n_params), 'epochs_run': len(hist), 'best_epoch': stop.best_epoch,
            'best_loss': stop.best, 'hist': hist, 'train_time_sec': round(time.time() - t0, 1), 'pca': pca, 'mods': mods}
    return Ltr, Lev, info


# ----------------------------------------------------------------------------- selection (validation only)
def select_job(job, data_dir, prefix, max_epochs):
    vi, rest = job // (N_SPLITS * N_FOLDS), job % (N_SPLITS * N_FOLDS)
    s, fold = rest // N_FOLDS, rest % N_FOLDS
    v = VARIANTS[vi]
    data = load_split(data_dir, prefix, s)
    X, y = data['Xtr'], data['ytr']
    rng = np.random.default_rng(1000 + 10 * s + fold)                       # same folds as the quantum selections
    idx_in = rng.permutation(np.where(y != fold)[0]); n_val = int(0.2 * len(idx_in))
    val_in, tr_in, val_out = idx_in[:n_val], idx_in[n_val:], np.where(y == fold)[0]
    keep = sorted(set(np.unique(y[tr_in]).tolist()))
    y_tr = np.array([keep.index(c) for c in y[tr_in]])
    X_val = np.vstack([X[val_in], X[val_out]])
    ood = np.r_[np.zeros(len(val_in), int), np.ones(len(val_out), int)]
    print(f'SELECTION job {job}: variant {vi} ({vname(v)}), split {s}, pseudo-OOD = known label {data["known_labels"][fold]}', flush=True)
    Ltr, (Lval,), info = train_embed(X[tr_in], y_tr, [X_val], SEL_SEED, max_epochs, v, tag='[sel] ')
    raw, mad = centroid_detect(Ltr, y_tr, Lval)
    ev = evaluate(raw, mad, ood)
    res = {'variant_index': vi, 'variant': v, 'split': s, 'fold': fold,
           'ROC_raw': ev['threshold_free']['ROC_AUC'], 'ROC_mad': ev['ROC_AUC_mad'],
           'CADE_F1@3.5': [e for e in ev['sweep'] if e['T_MAD'] == 3.5][0]['F1'],
           'best_epoch': info['best_epoch'], 'epochs_run': info['epochs_run'], 'n_params': info['n_params']}
    os.makedirs(SEL_DIR, exist_ok=True)
    json.dump(res, open(os.path.join(SEL_DIR, f'v{vi}_s{s}_f{fold}.json'), 'w'), indent=2)
    print(f'  VALIDATION  ROC_raw {res["ROC_raw"]:.4f}  ROC_mad {res["ROC_mad"]:.4f}  CADE_F1@3.5 {res["CADE_F1@3.5"]:.4f}', flush=True)


def select_summary():
    import glob
    rows = [json.load(open(f)) for f in glob.glob(os.path.join(SEL_DIR, 'v*_s*_f*.json'))]
    lines = []
    def log(s=''):
        print(s, flush=True); lines.append(s)
    log('=' * 110); log('CADE-mini -- VARIANT SELECTION on validation data (same folds and criterion as the quantum selections)')
    log('Criterion (fixed in advance): mean of (ROC_raw + ROC_mad) / 2 over the runs shared by all variants'); log('=' * 110)
    have = [vi for vi in range(len(VARIANTS)) if any(r['variant_index'] == vi for r in rows)]
    if not have:
        log('No selection results yet.'); return
    common = set.intersection(*[{(r['split'], r['fold']) for r in rows if r['variant_index'] == vi} for vi in have])
    missing = [vi for vi in range(len(VARIANTS)) if vi not in have]
    log(f'  Shared runs: {len(common)} of {N_SPLITS * N_FOLDS}' + (f'   (no results yet for variants {missing})' if missing else ''))
    if not common:
        return
    best, table = None, []
    log(f"\n  {'#':>2}  {'variant':<34}{'params':>7}{'ROC_raw':>10}{'ROC_mad':>10}{'CADE_F1@3.5':>13}{'epochs':>8}{'hit cap':>9}{'criterion':>11}")
    for vi in have:
        rs = [r for r in rows if r['variant_index'] == vi and (r['split'], r['fold']) in common]
        m = {k: float(np.mean([r[k] for r in rs])) for k in ['ROC_raw', 'ROC_mad', 'CADE_F1@3.5', 'epochs_run']}
        cap = sum(r['epochs_run'] >= SEL_EPOCHS for r in rs); crit = (m['ROC_raw'] + m['ROC_mad']) / 2
        table.append({'variant_index': vi, 'variant': VARIANTS[vi], 'runs': len(rs), **m, 'criterion': crit})
        log(f'  {vi:>2}  {vname(VARIANTS[vi]):<34}{rs[0]["n_params"]:>7}{m["ROC_raw"]:>10.4f}{m["ROC_mad"]:>10.4f}'
            f'{m["CADE_F1@3.5"]:>13.4f}{m["epochs_run"]:>8.0f}{cap:>5}/{len(rs):<3}{crit:>11.4f}')
        if best is None or crit > best['criterion']:
            best = table[-1]
    if missing or len(common) < N_SPLITS * N_FOLDS:
        log(f'\nNOTE: preliminary -- {len(common)} shared run(s) per variant; variants without results: {missing or "none"}.')
    log(f'\nSELECTED: variant {best["variant_index"]} -- {vname(best["variant"])}  (criterion {best["criterion"]:.4f})')
    json.dump({'selected': best, 'all': table, 'shared_runs': sorted(common)},
              open(os.path.join(SEL_DIR, 'selected_variant.json'), 'w'), indent=2)
    open(os.path.join(SEL_DIR, 'selection_summary.txt'), 'w').write('\n'.join(lines) + '\n')
    print(f'Saved: {SEL_DIR}/selected_variant.json, selection_summary.txt')


# ----------------------------------------------------------------------------- final runs
def load_variant(forced):
    if forced is not None:
        return forced, VARIANTS[forced]
    p = os.path.join(SEL_DIR, 'selected_variant.json')
    if not os.path.exists(p):
        raise SystemExit('No selection yet: run the selection array and "select-summary" first, or pass --variant N.')
    b = json.load(open(p))['selected']
    return b['variant_index'], VARIANTS[b['variant_index']]


def run(task_id, data_dir, prefix, results_dir, max_epochs, forced_variant=None):
    import torch
    vi, v = load_variant(forced_variant)
    run_id, split, seed = task_info(task_id)
    data = load_split(data_dir, prefix, split)
    name = f'CADE-mini selected [{vname(v)}]'
    print(f'{name} | task {task_id} run {run_id} split {split} seed {seed} OOD={data["family"]}', flush=True)
    Ltr, (Lte,), info = train_embed(data['Xtr'], data['ytr'], [data['Xte']], seed, max_epochs, v)
    raw, mad = centroid_detect(Ltr, data['ytr'], Lte)
    p2 = PCA(n_components=2).fit(Ltr)
    save_dir = os.path.join(results_dir, f'run{run_id}', f'split{split}'); os.makedirs(save_dir, exist_ok=True)
    meta = {'run_id': run_id, 'split_id': split, 'seed': seed, 'input': f'PCA-{PCA_DIM}', 'variant_index': vi, 'variant': v,
            'architecture': f'{v["design"]}: {DESIGNS[v["design"]]}', 'n_trainable_params': info['n_params'],
            'lr': v['lr'], 'epochs_run': info['epochs_run'], 'best_epoch': info['best_epoch'], 'best_loss': info['best_loss'],
            'patience': PATIENCE, 'train_time_sec': info['train_time_sec'],
            'pca_explained_variance': float(info['pca'].explained_variance_ratio_.sum())}
    torch.save(info['mods'].state_dict(), os.path.join(save_dir, 'model_weights.pt'))
    np.save(os.path.join(save_dir, 'train_anomaly_scores.npy'), centroid_detect(Ltr, data['ytr'], Ltr)[1])
    s = save_task(save_dir, name, meta, raw, mad, data, info['hist'], p2.transform(Ltr), p2.transform(Lte),
                  f'2-D PCA of the {Ltr.shape[1]}-D latent vectors')
    e = s['evaluation']
    print(f'  DONE  ROC_raw {e["threshold_free"]["ROC_AUC"]:.4f}  ROC_mad {e["ROC_AUC_mad"]:.4f}  '
          f'CADE_F1@3.5 {[x for x in e["sweep"] if x["T_MAD"] == 3.5][0]["F1"]:.4f}  -> {save_dir}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='CADE-mini with validation-based selection (matched to the quantum selections)')
    p.add_argument('cmd', help="'variants', 'select' (with --job), 'select-summary', a task 0-11, or 'aggregate'")
    p.add_argument('--job', type=int, help=f'selection job 0-{len(VARIANTS) * N_SPLITS * N_FOLDS - 1}')
    p.add_argument('--variant', type=int, default=None)
    p.add_argument('--data-dir', default=DEFAULT_DATA)
    p.add_argument('--prefix', default=DEFAULT_PREFIX)
    p.add_argument('--results-dir', default='results')
    p.add_argument('--max-epochs', type=int, default=MAX_EPOCHS)
    a = p.parse_args()
    if a.cmd == 'variants':
        for i, v in enumerate(VARIANTS):
            print(f'  {i:>2}: {vname(v)}')
    elif a.cmd == 'select':
        select_job(a.job, a.data_dir, a.prefix, min(a.max_epochs, SEL_EPOCHS))
    elif a.cmd == 'select-summary':
        select_summary()
    elif a.cmd == 'aggregate':
        rd = os.path.abspath(os.path.expanduser(a.results_dir))
        sel = os.path.exists(os.path.join(SEL_DIR, 'selected_variant.json')) or a.variant is not None
        aggregate(rd, f'CADE-mini selected [{vname(load_variant(a.variant)[1])}]' if sel else 'CADE-mini selected')
    else:
        run(int(a.cmd), a.data_dir, a.prefix, os.path.abspath(os.path.expanduser(a.results_dir)), a.max_epochs, a.variant)
