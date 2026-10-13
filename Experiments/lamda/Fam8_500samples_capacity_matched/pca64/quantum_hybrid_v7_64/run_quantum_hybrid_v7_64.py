#!/usr/bin/env python3
"""
run_quantum_hybrid_v7_64.py -- Fam8_500samples_capacity_matched/pca64/quantum_hybrid_v7_64   (standalone)
Quantum hybrid v7 at PCA-64: amplitude encoding of PCA-64 on 6 qubits + 11 angle re-uploads cycling all
64 PCs, 19 layers RX+RY+RZ + CNOT ring, re-upload scale init 0.1 (near identity, as amp-64). 474 params (amp-64 budget).
Experiment: 8-family LAMDA, month-stratified downsized to max 500 samples/family (cade_lamda_500_vt), PCA-64 input.
24 tasks = 8 held-out families x 3 seeds. Shared code, losses, training and scoring identical to quantum_hybrid_v7.
Usage (from Fam8_500samples_capacity_matched/pca64/quantum_hybrid_v7_64/):
  python3 run_quantum_hybrid_v7_64.py probe        # optional pre-check (weight dependence); not required
  mkdir -p logs && sbatch run_quantum_hybrid_v7_64.slurm     # 24 test tasks
  python3 run_quantum_hybrid_v7_64.py aggregate
"""
# =============================================================================
# SHARED CODE -- identical in all capacity_matched scripts of this experiment
# (data, PCA, CADE pairing, early stopping, MAD detection, metrics,
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
PCA_DIM        = 64
THRESHOLDS     = [0.0, 2.0, 2.5, 3.0, 3.5]
MAD_T          = 3.5
MAD_B          = 1.4826
BATCH_SIZE     = 64
SIMILAR_RATIO  = 0.25
MAX_EPOCHS     = 250
PATIENCE       = 20
REL_MIN_DELTA  = 1e-4
LR             = 0.001
DEFAULT_DATA   = '~/projects/OOD/Data/cade_lamda_500_vt'
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



import argparse

# ============================================================================= MODEL-SPECIFIC CODE
INIT_STD   = 0.1
MARGIN     = 1.0
TAU        = 0.1                    # temperature of the prototype loss
LAMBDA_P   = 1.0                    # weight of the prototype loss


def _cnot_perm(n, pairs):
    """Index map for a fixed sequence of CNOTs (control, target) on n qubits (wire 0 = most significant bit)."""
    perm = np.arange(2 ** n)
    for c, t in pairs:
        bits = (perm[:, None] >> (n - 1 - np.arange(n))) & 1
        new = perm.copy(); flip = bits[:, c] == 1
        new[flip] = perm[flip] ^ (1 << (n - 1 - t)); perm = new
    inv = np.empty_like(perm); inv[perm] = np.arange(2 ** n)
    return inv


class Sim:
    """Exact batched statevector simulator (PyTorch, differentiable)."""
    def __init__(self, n, extra_pairs=()):
        import torch
        self.n = n
        self.ring = torch.from_numpy(_cnot_perm(n, [(k, k + 1) for k in range(n - 1)] + [(n - 1, 0)])).long()
        self.extra = torch.from_numpy(_cnot_perm(n, list(extra_pairs))).long() if extra_pairs else None

    def rot(self, psi, th, gates):
        """Single-qubit rotations on every qubit; gates = 'xyz' (RX, RY, RZ) or 'yz' (RY, RZ), applied in order."""
        import torch
        B = psi.shape[0]; t = psi.reshape([B] + [2] * self.n)
        for k in range(self.n):
            U = torch.eye(2, dtype=torch.complex64)
            for j, g in enumerate(gates):
                a = th[k, j]; c, s = torch.cos(a / 2), torch.sin(a / 2)
                if g == 'x':
                    G = torch.stack([torch.stack([c + 0j, -1j * s]), torch.stack([-1j * s, c + 0j])])
                elif g == 'y':
                    G = torch.stack([torch.stack([c + 0j, -s + 0j]), torch.stack([s + 0j, c + 0j])])
                else:
                    e = torch.exp(-0.5j * a); G = torch.stack([torch.stack([e, 0 * e]), torch.stack([0 * e, e.conj()])])
                U = G.to(torch.complex64) @ U
            t = torch.movedim(torch.tensordot(t, U.T.to(psi.dtype), dims=([k + 1], [0])), -1, k + 1)
        return t.reshape(B, -1)

    def cnot_ring(self, psi):
        return psi[:, self.ring]

    def cnot_extra(self, psi):
        return psi[:, self.extra]


def angle_upload(psi, ang, a, b, n):
    """RY(a_k * ang_k + b_k) on qubit k, per sample."""
    import torch
    B = psi.shape[0]; t = psi.reshape([B] + [2] * n); shape = [B] + [1] * (n - 1)
    th = ang * a + b
    for k in range(n):
        c = torch.cos(th[:, k] / 2).to(psi.dtype).view(shape); s_ = torch.sin(th[:, k] / 2).to(psi.dtype).view(shape)
        t0, t1 = t.select(k + 1, 0), t.select(k + 1, 1)
        t = torch.stack([c * t0 - s_ * t1, s_ * t0 + c * t1], dim=k + 1)
    return t.reshape(B, -1)


class HybInput:
    """PCA-64 (training data only) -> amplitudes (L2-normalized PCA-64) and angles (center, ONE shared scale = std of
    PC1, arctan -- exactly the angle-8 preprocessing). Output columns: [8 amplitudes | 8 angles]."""
    def fit(self, Xtr):
        self.pca = PCA(n_components=PCA_DIM, svd_solver='full').fit(Xtr)
        Z = self.pca.transform(Xtr); self.mu, self.scale = Z.mean(0), float(Z[:, 0].std()) + 1e-8
        return self

    def __call__(self, X):
        Z = self.pca.transform(X)
        amp = (Z / np.linalg.norm(Z, axis=1, keepdims=True).clip(min=1e-8)).astype(np.float32)
        ang = np.arctan((Z - self.mu) / self.scale).astype(np.float32)
        return np.hstack([amp, ang])


def pure_prototypes(S, y, C):
    """Dominant eigenvector of each class's density matrix (a pure state, phase-invariant)."""
    import torch
    St = torch.as_tensor(S); V = []
    for c in range(C):
        P = St[torch.as_tensor(y) == c]; rho = P.T @ P.conj() / len(P)
        V.append(torch.linalg.eigh(rho)[1][:, -1].conj())
    return torch.stack(V)


# ----------------------------------------------------------------------------- Quantum hybrid v7 at PCA-64 (6 qubits)
# Hybrid v7's design on 6 qubits: PCA-64 amplitude-encoded (the full 2^6 state), then 11 learnable angle re-uploads of
# 6 PCs each, cycling through all 64 PCs (each PC once, PCs 1-2 twice), spread over 19 layers RX+RY+RZ + CNOT ring;
# pure-state prototypes; trained like v7 (pair contrastive + prototype cross-entropy).
#   19 x 6 x 3 = 342 + 11 uploads x 6 x 2 = 132  -> 474 params (amp-64's budget).
# Re-upload scale init a = 0.1 (near identity), as in amp-64: at 64 amplitudes, full-strength re-uploads from the start
# scrambled the input's cos^2 geometry in the PCA-256 pilot; starting near identity starts AT that geometry.
N_Q, L_Q, Q_INIT = 6, 19, 0.1
UP_Q = (1, 3, 4, 6, 7, 9, 11, 12, 14, 15, 17)
_seq = [p % PCA_DIM for p in range(6 * 11)]
GROUPS = [tuple(_seq[6 * u:6 * u + 6]) for u in range(11)]
assert len(UP_Q) == len(GROUPS) == 11 and max(UP_Q) < L_Q
NAME = 'Quantum hybrid v7 at PCA-64 (6 qubits, 19 layers, 11 re-uploads cycling all 64 PCs, 474 params)'
INPUT_DOC = 'PCA-64: L2-normalized amplitudes + arctan(shared-scale PCs) angles re-uploaded 11 times in groups of 6'
ARCH_DOC = (f'{N_Q} qubits amplitude, {L_Q} layers RX+RY+RZ+ring, 11 angle re-uploads at {UP_Q} cycling all 64 PCs '
            f'(trainable scaling, init a={Q_INIT}), pure-state prototypes')
_SIM = {}


def init_params():
    import torch
    return {'W': torch.nn.Parameter(torch.randn(L_Q, N_Q, 3) * INIT_STD),
            'a': torch.nn.Parameter(torch.full((len(UP_Q), N_Q), Q_INIT)), 'b': torch.nn.Parameter(torch.zeros(len(UP_Q), N_Q))}


def forward_regs(A, P):
    import torch
    if not _SIM:
        _SIM['s'] = Sim(N_Q)
    sim = _SIM['s']; amp, ang = A[:, :PCA_DIM], A[:, PCA_DIM:]
    psi = amp.to(torch.complex64); u = 0
    for l in range(L_Q):
        if l in UP_Q:
            psi = angle_upload(psi, ang[:, list(GROUPS[u])], P['a'][u], P['b'][u], N_Q); u += 1
        psi = sim.cnot_ring(sim.rot(psi, P['W'][l], 'xyz'))
    return [psi]


# ----------------------------------------------------------------------------- multi-register fidelity engine
# A model maps a sample to one state per register; the fidelity between two samples is the PRODUCT of the
# per-register fidelities (registers are never entangled with each other), and each family's prototype is one
# pure state per register. With a single register this reduces to the usual fidelity / pure-prototype model.
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


def train_embed(Xtr, ytr, X_eval, seed, max_epochs, tag=''):
    import torch, torch.optim as optim
    inp = HybInput().fit(Xtr); Atr = inp(Xtr); C = len(np.unique(ytr))
    set_seed(seed); P = init_params(); params = list(P.values())
    n_params = int(sum(p.numel() for p in params)); opt = optim.Adam(params, lr=LR)
    st = lambda A: forward_regs(A, P)
    def all_states():
        with torch.no_grad():
            outs = [st(torch.from_numpy(Atr[i:i + 512])) for i in range(0, len(Atr), 512)]
            return [torch.cat([o[r] for o in outs]) for r in range(len(outs[0]))]
    print(f'  {tag}{NAME}: {n_params} trainable params', flush=True)
    stop, hist, t0 = EarlyStopper(), [], time.time()
    snap = lambda: {k: v.detach().clone() for k, v in P.items()}
    stop.state = snap()
    for epoch in range(1, max_epochs + 1):
        Vs = [pure_prototypes(S, ytr, C).detach() for S in all_states()]   # prototypes refreshed every epoch
        bc, bx, by = epoch_batches(Atr, ytr); ls = []
        for b in range(bc):
            A, yb = torch.from_numpy(bx[b]), torch.from_numpy(by[b])
            opt.zero_grad()
            Ss = st(A); h = Ss[0].shape[0] // 2
            Fp, Fproto = 1.0, 1.0
            for S, V in zip(Ss, Vs):
                ov = (S[:h].conj() * S[h:]).sum(-1); Fp = Fp * (ov.real ** 2 + ov.imag ** 2)
                Fproto = Fproto * (S @ V.T).abs() ** 2
            d = 1.0 - Fp; same = (yb[:h] == yb[h:]).float()
            pair = (same * d + (1 - same) * torch.relu(MARGIN - d)).mean()
            proto = torch.nn.functional.cross_entropy(Fproto / TAU, yb)
            loss = pair + LAMBDA_P * proto
            loss.backward(); opt.step(); ls.append(loss.item())
        hist.append(float(np.mean(ls)))
        done = stop.step(epoch, hist[-1], snap)
        print(f'  {tag}epoch {epoch:3d} loss {hist[-1]:.5f} best {stop.best:.5f} (ep {stop.best_epoch}) '
              f'no_improve {stop.since} ({time.time() - t0:.0f}s)', flush=True)
        if done:
            break
    with torch.no_grad():
        for k in P:
            P[k].copy_(stop.state[k])
        def F(X):
            outs = [st(torch.from_numpy(inp(X[i:i + 512]))) for i in range(0, len(X), 512)]
            return [np.vstack([o[r].numpy() for o in outs]) for r in range(len(outs[0]))]
        Ftr, Fev = F(Xtr), [F(X) for X in X_eval]
    info = {'n_params': n_params, 'epochs_run': len(hist), 'best_epoch': stop.best_epoch,
            'best_loss': (None if not np.isfinite(stop.best) else stop.best), 'hist': hist,
            'train_time_sec': round(time.time() - t0, 1), 'inp': inp, 'P': {k: v.detach() for k, v in P.items()}}
    return Ftr, Fev, info


def run(task_id, data_dir, prefix, results_dir, max_epochs):
    import torch
    from sklearn.decomposition import KernelPCA
    run_id, split, seed = task_info(task_id)
    data = load_split(data_dir, prefix, split)
    print(f'{NAME} | task {task_id} run {run_id} split {split} seed {seed} OOD={data["family"]}', flush=True)
    Ftr, (Fte,), info = train_embed(data['Xtr'], data['ytr'], [data['Xte']], seed, max_epochs)
    raw, mad = proto_detect(Ftr, data['ytr'], Fte)
    sub = np.random.default_rng(0).choice(len(Ftr[0]), min(800, len(Ftr[0])), replace=False)
    Fsub = [S[sub] for S in Ftr]
    kp = KernelPCA(n_components=2, kernel='precomputed').fit(prod_fid(Fsub, Fsub))
    save_dir = os.path.join(results_dir, f'run{run_id}', f'split{split}'); os.makedirs(save_dir, exist_ok=True)
    meta = {'run_id': run_id, 'split_id': split, 'seed': seed, 'input': INPUT_DOC, 'architecture': ARCH_DOC,
            'pca_explained_variance': float(info['inp'].pca.explained_variance_ratio_.sum()),
            'n_trainable_params': info['n_params'], 'lr': LR, 'epochs_run': info['epochs_run'],
            'best_epoch': info['best_epoch'], 'best_loss': info['best_loss'], 'patience': PATIENCE,
            'train_time_sec': info['train_time_sec']}
    torch.save(info['P'], os.path.join(save_dir, 'model_weights.pt'))
    arrs = {}
    for r in range(len(Ftr)):
        arrs[f'states_train_r{r}'] = Ftr[r]; arrs[f'states_test_r{r}'] = Fte[r]
    np.savez_compressed(os.path.join(save_dir, 'states.npz'), y_train=data['ytr'], y_test=data['yte'], **arrs)
    tr_raw, tr_A = proto_detect(Ftr, data['ytr'], Ftr)
    np.save(os.path.join(save_dir, 'train_anomaly_scores.npy'), tr_A)
    np.save(os.path.join(save_dir, 'train_raw_scores.npy'), tr_raw)
    s = save_task(save_dir, NAME, meta, raw, mad, data, info['hist'],
                  kp.transform(prod_fid(Ftr, Fsub)), kp.transform(prod_fid(Fte, Fsub)), 'kernel PCA (product fidelity kernel)')
    e = s['evaluation']
    print(f'  DONE  ROC_raw {e["threshold_free"]["ROC_AUC"]:.4f}  ROC_mad {e["ROC_AUC_mad"]:.4f}  '
          f'CADE_F1@3.5 {[x for x in e["sweep"] if x["T_MAD"] == 3.5][0]["F1"]:.4f}  -> {save_dir}', flush=True)


def probe(data_dir, prefix, n=400):
    """Pre-check, no training: per-register and product fidelities (same vs different family) for two random
    weight draws, the input cos^2 of the PCA-64 amplitudes, and the random-state floor."""
    import torch
    for s in range(N_SPLITS):
        data = load_split(data_dir, prefix, s)
        idx = np.random.default_rng(0).choice(len(data['ytr']), min(n, len(data['ytr'])), replace=False); y = data['ytr'][idx]
        A = torch.from_numpy(HybInput().fit(data['Xtr'])(data['Xtr'][idx]))
        with torch.no_grad():
            set_seed(1); S1 = [x.numpy() for x in forward_regs(A, init_params())]
            set_seed(2); S2 = [x.numpy() for x in forward_regs(A, init_params())]
        same = (y[:, None] == y[None, :]) & ~np.eye(len(y), dtype=bool); diff = y[:, None] != y[None, :]; off = ~np.eye(len(y), dtype=bool)
        parts = []
        for r, (a, b) in enumerate(zip(S1, S2)):
            F = np.abs(a.conj() @ a.T) ** 2
            parts.append(f'reg{r} same {F[same].mean():.3f} diff {F[diff].mean():.3f} (floor {1 / a.shape[1]:.4f})')
        P1, P2 = prod_fid(S1, S1), prod_fid(S2, S2); amp = A[:, :PCA_DIM].numpy(); C2 = (amp @ amp.T) ** 2
        print(f'  split{s} ({data["family"]:<12}) ' + ' | '.join(parts) +
              f' | product same {P1[same].mean():.3f} diff {P1[diff].mean():.3f} | input cos^2 same {C2[same].mean():.3f} '
              f'diff {C2[diff].mean():.3f} | weight dependence {np.abs(P1 - P2)[off].mean():.4f}')
    print('Healthy: same-family fidelity > different-family fidelity, both above the floor, weight dependence > 0.')


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'probe':
        ap = argparse.ArgumentParser(); ap.add_argument('cmd')
        ap.add_argument('--data-dir', default=DEFAULT_DATA); ap.add_argument('--prefix', default=DEFAULT_PREFIX)
        a = ap.parse_args(); probe(a.data_dir, a.prefix)
    else:
        standard_main(f'PCA-64 experiment: {NAME}', run, NAME)
