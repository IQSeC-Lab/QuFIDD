#!/usr/bin/env python3
"""
compare_all.py -- Full_new_experiments (top level)   (standalone script)
=======================================================================
Pairs the 12 tasks of pca_only, cade_mini_full, cade_mini_encoder_only and quantum
by (run, split) and reports, for every metric: mean +- std, per-task wins, two-sided
Wilcoxon signed-rank p-values, per-family tables, and comparison plots.
[+ quantum/results_untrained if the untrained ablation was run]

THRESHOLD CALIBRATION (is T_MAD = 3.5 equally strict for every model?)
  For every model and task, the anomaly scores A of the TRAINING samples (known families
  only; never the OOD family) are recomputed from the saved models/states, and:
    FPR_known@3.5    share of KNOWN test samples flagged at T = 3.5 (false-alarm rate)
    TrainExceed@3.5  share of TRAINING samples with A > 3.5
    T_cal            per-task threshold = the (1 - target) quantile of the training A,
                     i.e. the T at which a fixed share (default 5%) of known training
                     samples would be flagged -- the same false-alarm budget for all models
    FPR_known@Tcal   realized false-alarm rate on known test samples at T_cal
    F1fixed@Tcal     fixed-threshold F1 at T_cal (flag A > T_cal)
    CADE_F1@Tcal     CADE protocol at T_cal (flag A > T_cal, rank by distance, best F1)
  Needs the data + saved weights/states (PyTorch for the CADE-minis); nothing is retrained.
  Skip with --no-calibration.

STRUCTURE-AWARE STATISTICS (the 12 tasks = 4 families x 3 seeds are not independent)
  (1) per-task DeLong tests on ROC-AUC (raw and MAD score), same test samples
  (2) per-task paired bootstrap over test samples for CADE F1 @3.5 and threshold-free F1
  (3) hierarchical bootstrap (families -> seeds -> test samples): one 95% CI of the mean difference
  Outputs delong_per_task.csv, bootstrap_per_task.csv, hierarchical_bootstrap.csv and
  plots/hierarchical_bootstrap_forest.png. Key pairs by default (--stats-pairs all for every pair).

Usage (from Full_new_experiments/, after every model has finished and aggregated):
  python3 compare_all.py                       # includes calibration (target 5%)
  python3 compare_all.py --target-fpr 0.10
  python3 compare_all.py --no-calibration
Outputs (Full_new_experiments/comparison/): comparison_summary.txt,
comparison_per_task.csv, plots/*.png
"""

# =============================================================================
# SHARED CODE -- identical to the four model scripts (metric keys, aggregation)
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
# COMPARISON
# =============================================================================

import os, sys, csv
import numpy as np
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(HERE, 'comparison')
FE = HERE                                   # this script sits in Full_new_experiments/
MODELS = [('PCA-5 only', 'pca_only/results'),
          ('PCA-5 only (L2)', 'pca_only/results_l2'),
          ('CADE-mini full', 'cade_mini_full/results'),
          ('CADE-mini encoder-only', 'cade_mini_encoder_only/results'),
          ('Quantum', 'quantum/results'),
          ('Quantum untrained', 'quantum/results_untrained'),
          ('Quantum v2', 'quantum_v2/results'),
          ('Quantum v3', 'quantum_v3/results'),
          ('CADE-mini selected', 'cade_mini_select/results')]
BASE = 'PCA-5 only (L2)'                    # primary zero-parameter baseline (stronger than raw coordinates)
TESTS = [('Quantum', 'CADE-mini full'), ('Quantum', 'CADE-mini encoder-only'), ('Quantum', BASE),
         ('Quantum', 'Quantum untrained'), ('CADE-mini encoder-only', 'CADE-mini full'),
         ('CADE-mini full', BASE), ('CADE-mini encoder-only', BASE),
         ('Quantum v2', 'CADE-mini full'), ('Quantum v2', 'CADE-mini encoder-only'), ('Quantum v2', BASE),
         ('Quantum v2', 'Quantum'),
         ('Quantum v3', 'Quantum v2'), ('Quantum v3', 'CADE-mini selected'), ('Quantum v2', 'CADE-mini selected'),
         ('Quantum v3', 'CADE-mini full'), ('Quantum v3', 'CADE-mini encoder-only'), ('Quantum v3', BASE),
         ('CADE-mini selected', 'CADE-mini full'), ('CADE-mini selected', 'CADE-mini encoder-only'),
         ('CADE-mini selected', BASE)]
MAIN = ['TF_F1', 'ROC_raw', 'PR_raw', 'ROC_mad', 'MAD_F1_fixed'] + [f'F1@T{T}' for T in THRESHOLDS[1:]]
CAL = ['CADE_F1@Tcal', 'F1fixed@Tcal', 'FPR_known@Tcal', 'T_cal', 'FPR_known@3.5', 'TrainExceed@3.5']
LABEL = {'TF_F1': 'F1 threshold-free', 'ROC_raw': 'ROC-AUC (raw)', 'PR_raw': 'PR-AUC (raw)',
         'ROC_mad': 'ROC-AUC (MAD score)', 'MAD_F1_fixed': 'MAD F1 fixed T=3.5', 'F1@T3.5': 'F1 CADE protocol T=3.5',
         'CADE_F1@Tcal': 'F1 CADE protocol T=T_cal', 'F1fixed@Tcal': 'MAD F1 fixed T=T_cal',
         'FPR_known@Tcal': 'known-test FPR at T_cal', 'T_cal': 'calibrated T_cal',
         'FPR_known@3.5': 'known-test FPR at 3.5', 'TrainExceed@3.5': 'train share A>3.5'}
REL = dict(MODELS)
ENC = {'CADE-mini full': ([11], True), 'CADE-mini encoder-only': ([22], False)}


# ------------------------------------------------------------------ threshold calibration
def _mad_A(dists_train_own, D_eval):
    """dists_train_own[c]: training distances of class-c samples to their own prototype/centroid;
    D_eval: [n, C] distances of evaluated samples to every class. Returns A = min_c |d - med_c| / MAD_c."""
    A = []
    for c, d in enumerate(dists_train_own):
        med, mad = _mad_stats(d)
        A.append(np.abs(D_eval[:, c] - med) / mad)
    return np.min(A, axis=0)


def train_scores(name, r, s, data, pca):
    """Anomaly scores A of the TRAINING samples, recomputed exactly as at test time."""
    ytr = data['ytr']; C = len(np.unique(ytr))
    tdir = os.path.join(FE, REL[name], f'run{r}', f'split{s}')
    if os.path.exists(os.path.join(tdir, 'train_anomaly_scores.npy')):          # saved by quantum_v2
        return np.load(os.path.join(tdir, 'train_anomaly_scores.npy'))
    if name.startswith('Quantum'):
        St = np.load(os.path.join(tdir, 'states.npz'))['states_train']
        own, D = [], np.zeros((len(ytr), C))
        for c in range(C):
            m = ytr == c; P = St[m]; n = m.sum()
            F = np.abs(St.conj() @ P.T) ** 2                    # [N, n_c]
            Dc = 1.0 - F.mean(axis=1)
            Dc[m] = 1.0 - (F[m].sum(axis=1) - 1.0) / (n - 1)     # leave-one-out for own-class samples
            D[:, c] = Dc; own.append(Dc[m])
        return _mad_A(own, D)
    Z = pca.transform(data['Xtr'])
    if name.startswith('PCA-5'):
        L = Z / np.linalg.norm(Z, axis=1, keepdims=True).clip(min=1e-8) if '(L2)' in name else Z
    else:
        import torch, torch.nn as nn
        hid, dec = ENC[name]; dims = [PCA_DIM] + hid + [C]
        enc = []
        for i in range(len(dims) - 2):
            enc += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
        enc += [nn.Linear(dims[-2], dims[-1])]
        mods = [nn.Sequential(*enc)]
        if dec:
            d_ = []
            for i in range(len(dims) - 1, 1, -1):
                d_ += [nn.Linear(dims[i], dims[i - 1]), nn.ReLU()]
            d_ += [nn.Linear(dims[1], dims[0])]
            mods.append(nn.Sequential(*d_))
        ml = nn.ModuleList(mods); ml.load_state_dict(torch.load(os.path.join(tdir, 'model_weights.pt'), map_location='cpu'))
        with torch.no_grad():
            L = ml[0](torch.from_numpy(Z.astype(np.float32))).numpy()
    cents = [L[ytr == c].mean(axis=0) for c in range(C)]
    D = np.stack([np.linalg.norm(L - mu, axis=1) for mu in cents], axis=1)
    return _mad_A([D[ytr == c, c] for c in range(C)], D)


def calibrate(data_all, names, keys, data_dir, prefix, target, log):
    """Adds CAL metrics to data_all[name][key] for every model/task that can be recomputed."""
    cache = {}
    for k in keys:
        r, s = k
        if s not in cache:
            d = load_split(data_dir, prefix, s); cache[s] = (d, fit_pca(d['Xtr']))
        d, pca = cache[s]
        for n in names:
            if k not in data_all[n]:
                continue
            try:
                At = train_scores(n, r, s, d, pca)
            except Exception as e:
                log(f'  calibration skipped for {n} run{r}/split{s}: {e}'); continue
            sc = np.genfromtxt(os.path.join(FE, REL[n], f'run{r}', f'split{s}', 'ood_scores.csv'), delimiter=',', names=True)
            raw, mad, ood = sc['raw_score'], sc['mad_score'], sc['is_ood'].astype(int)
            known = ood == 0
            T = float(np.quantile(At, 1.0 - target))
            flag = mad > T; act = ood.astype(bool)
            tp, fp, fn = int((flag & act).sum()), int((flag & ~act).sum()), int((~flag & act).sum())
            P = tp / (tp + fp) if tp + fp else 0.0; R = tp / (tp + fn) if tp + fn else 0.0
            data_all[n][k].update({
                'T_cal': T, 'FPR_known@Tcal': float(flag[known].mean()),
                'F1fixed@Tcal': (2 * P * R / (P + R) if P + R else 0.0),
                'CADE_F1@Tcal': _best_down_list(raw, ood, np.where(flag)[0])['F1'],
                'FPR_known@3.5': float((mad[known] > MAD_T).mean()),
                'TrainExceed@3.5': float((At > MAD_T).mean())})


# ------------------------------------------------------------------ statistics that respect the data structure
# The 12 tasks are 4 held-out families x 3 seeds on the SAME data, so they are not independent.
# Three levels of evidence:
#   (1) per-task DeLong test on ROC-AUC (both models scored on the same test samples)
#   (2) per-task paired bootstrap over test samples for F1 metrics (CI + p of the difference)
#   (3) hierarchical bootstrap: resample families, then seeds within family, then test samples ->
#       one 95% CI for the mean difference that accounts for all three sources of variation
from scipy.stats import norm as _norm, rankdata as _rankdata
KEY_PAIRS = [('Quantum v2', 'CADE-mini selected'), ('Quantum v3', 'CADE-mini selected'), ('Quantum v3', 'Quantum v2'),
             ('Quantum v2', 'CADE-mini full'), ('Quantum v2', 'CADE-mini encoder-only'), ('Quantum v2', 'Quantum'),
             ('Quantum v2', BASE), ('CADE-mini selected', BASE)]
STAT_METRICS = ['ROC_raw', 'ROC_mad', 'CADE_F1@3.5', 'TF_F1']


def _auc(s, y):
    r = _rankdata(s); n1 = int(y.sum()); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def _best_f1(raw, y, flag=None):
    idx = np.arange(len(raw)) if flag is None else np.where(flag)[0]
    if len(idx) == 0 or y.sum() == 0:
        return 0.0
    o = idx[np.argsort(-raw[idx], kind='stable')]
    tp = np.cumsum(y[o]); k = np.arange(1, len(o) + 1)
    p, r = tp / k, tp / y.sum()
    return float(np.max(np.where(p + r > 0, 2 * p * r / (p + r + 1e-12), 0.0)))


def _metric(m, raw, mad, y):
    if m == 'ROC_raw':
        return _auc(raw, y)
    if m == 'ROC_mad':
        return _auc(mad, y)
    if m == 'CADE_F1@3.5':
        return _best_f1(raw, y, mad > MAD_T)
    return _best_f1(raw, y)                                   # TF_F1


def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1; i = j
    T2 = np.empty(N); T2[J] = T
    return T2


def delong(y, s1, s2):
    """Two-sided DeLong test for AUC(s1) vs AUC(s2) on the same samples -> (auc1, auc2, z, p)."""
    m, n = int((y == 1).sum()), int((y == 0).sum())
    aucs, V10, V01 = [], [], []
    for s_ in (s1, s2):
        P, Q = s_[y == 1], s_[y == 0]
        tx, ty, tz = _midrank(P), _midrank(Q), _midrank(np.concatenate([P, Q]))
        aucs.append(tz[:m].sum() / (m * n) - (m + 1.0) / (2.0 * n))
        V10.append((tz[:m] - tx) / n); V01.append(1.0 - (tz[m:] - ty) / m)
    S = np.cov(np.array(V10)) / m + np.cov(np.array(V01)) / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    if var <= 0:
        return aucs[0], aucs[1], 0.0, 1.0
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return aucs[0], aucs[1], z, float(2 * _norm.sf(abs(z)))


def _scores(name, k):
    p = os.path.join(FE, REL[name], f'run{k[0]}', f'split{k[1]}', 'ood_scores.csv')
    a = np.genfromtxt(p, delimiter=',', names=True)
    return a['raw_score'], a['mad_score'], a['is_ood'].astype(int)


def run_statistics(data, names, fam_of, n_boot, pairs, log, outd, seed=0):
    rng = np.random.default_rng(seed)
    rows_delong, rows_boot, rows_hier = [], [], []
    for A, B in pairs:
        if A not in data or B not in data:
            continue
        keys = sorted(set(data[A]) & set(data[B]))
        if not keys:
            continue
        S = {k: (_scores(A, k), _scores(B, k)) for k in keys}
        fams = sorted({fam_of[k] for k in keys})
        log(f'\n  {A}  vs  {B}   ({len(keys)} paired tasks, {len(fams)} families)')
        # (1) DeLong per task
        for sc, nm in [(0, 'ROC_raw'), (1, 'ROC_mad')]:
            per_fam = {}
            for k in keys:
                (a, b) = S[k]; y = a[2]
                a1, a2, z, p = delong(y, a[sc], b[sc])
                rows_delong.append([A, B, nm, k[0], k[1], fam_of[k], a1, a2, a1 - a2, z, p])
                per_fam.setdefault(fam_of[k], []).append('A' if (p < 0.05 and a1 > a2) else 'B' if (p < 0.05 and a2 > a1) else '-')
            sa = sum(v.count('A') for v in per_fam.values()); sb = sum(v.count('B') for v in per_fam.values())
            fa = sum(all(x == 'A' for x in v) for v in per_fam.values()); fb = sum(all(x == 'B' for x in v) for v in per_fam.values())
            detail = '  '.join(f'{f}:{"".join(per_fam[f])}' for f in fams)
            log(f'    DeLong {nm:<8} tasks sig A>B: {sa:>2} | B>A: {sb:>2} | n.s.: {len(keys) - sa - sb:>2}   '
                f'families with every seed sig: A {fa}, B {fb}   [{detail}]')
        # (2) per-task paired bootstrap for F1 metrics
        for nm in ['CADE_F1@3.5', 'TF_F1']:
            per_fam = {}
            for k in keys:
                (a, b) = S[k]; y = a[2]; n = len(y)
                d0 = _metric(nm, a[0], a[1], y) - _metric(nm, b[0], b[1], y)
                ds = np.empty(n_boot)
                for t in range(n_boot):
                    i = rng.integers(0, n, n)
                    ds[t] = _metric(nm, a[0][i], a[1][i], y[i]) - _metric(nm, b[0][i], b[1][i], y[i])
                lo, hi = np.percentile(ds, [2.5, 97.5]); p = min(1.0, 2 * min((ds <= 0).mean(), (ds >= 0).mean()))
                rows_boot.append([A, B, nm, k[0], k[1], fam_of[k], d0, lo, hi, p])
                per_fam.setdefault(fam_of[k], []).append('A' if lo > 0 else 'B' if hi < 0 else '-')
            sa = sum(v.count('A') for v in per_fam.values()); sb = sum(v.count('B') for v in per_fam.values())
            detail = '  '.join(f'{f}:{"".join(per_fam[f])}' for f in fams)
            log(f'    Boot   {nm:<11} tasks with 95% CI above 0: {sa:>2} | below 0: {sb:>2} | incl. 0: {len(keys) - sa - sb:>2}   [{detail}]')
        # (3) hierarchical bootstrap for the mean difference
        by_fam = {f: [k for k in keys if fam_of[k] == f] for f in fams}
        for nm in STAT_METRICS:
            point = float(np.mean([_metric(nm, *S[k][0]) - _metric(nm, *S[k][1]) for k in keys]))
            reps = np.empty(n_boot)
            for t in range(n_boot):
                diffs = []
                for f in rng.choice(fams, len(fams)):
                    for k in rng.choice(len(by_fam[f]), len(by_fam[f])):
                        kk = by_fam[f][k]; (a, b) = S[kk]; y = a[2]; i = rng.integers(0, len(y), len(y))
                        diffs.append(_metric(nm, a[0][i], a[1][i], y[i]) - _metric(nm, b[0][i], b[1][i], y[i]))
                reps[t] = np.mean(diffs)
            lo, hi = np.percentile(reps, [2.5, 97.5])
            verdict = 'A better' if lo > 0 else 'B better' if hi < 0 else 'no clear difference'
            rows_hier.append([A, B, nm, point, lo, hi, verdict, len(fams), len(keys)])
            log(f'    Hierarchical bootstrap {nm:<11} mean diff {point:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]   -> {verdict}')
    with open(os.path.join(outd, 'delong_per_task.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['A', 'B', 'metric', 'run', 'split', 'family', 'AUC_A', 'AUC_B', 'diff', 'z', 'p']); w.writerows(rows_delong)
    with open(os.path.join(outd, 'bootstrap_per_task.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['A', 'B', 'metric', 'run', 'split', 'family', 'diff', 'ci_lo', 'ci_hi', 'p']); w.writerows(rows_boot)
    with open(os.path.join(outd, 'hierarchical_bootstrap.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['A', 'B', 'metric', 'mean_diff', 'ci_lo', 'ci_hi', 'verdict', 'families', 'tasks']); w.writerows(rows_hier)
    # forest plot of the hierarchical CIs
    if rows_hier:
        pdir = os.path.join(outd, 'plots'); os.makedirs(pdir, exist_ok=True)
        fig, ax = plt.subplots(1, len(STAT_METRICS), figsize=(4.2 * len(STAT_METRICS), 0.5 * (len(rows_hier) // len(STAT_METRICS)) + 2.2), sharey=True)
        labels = list(dict.fromkeys(f'{r[0]} vs {r[1]}' for r in rows_hier))
        for a_, nm in zip(np.atleast_1d(ax), STAT_METRICS):
            rs = [r for r in rows_hier if r[2] == nm]
            yy = [labels.index(f'{r[0]} vs {r[1]}') for r in rs]
            a_.errorbar([r[3] for r in rs], yy, xerr=[[r[3] - r[4] for r in rs], [r[5] - r[3] for r in rs]], fmt='o', capsize=3)
            a_.axvline(0, color='k', lw=1); a_.set_title(nm, fontsize=10); a_.grid(alpha=0.3)
            a_.set_yticks(range(len(labels))); a_.set_yticklabels(labels, fontsize=7); a_.invert_yaxis()
        fig.suptitle('Hierarchical bootstrap: mean difference (A minus B) with 95% CI -- families, seeds, test samples resampled', fontsize=10)
        fig.tight_layout(); fig.savefig(os.path.join(pdir, 'hierarchical_bootstrap_forest.png'), dpi=140); plt.close(fig)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default=DEFAULT_DATA)
    ap.add_argument('--prefix', default=DEFAULT_PREFIX)
    ap.add_argument('--target-fpr', type=float, default=0.05, help='false-alarm budget on known training samples')
    ap.add_argument('--no-calibration', action='store_true')
    ap.add_argument('--n-boot', type=int, default=1000, help='bootstrap replicates (per-task and hierarchical)')
    ap.add_argument('--stats-pairs', choices=['key', 'all'], default='key', help="pairs for DeLong/bootstrap statistics")
    ap.add_argument('--no-stats', action='store_true', help='skip DeLong / bootstrap statistics')
    args = ap.parse_args()
    data = {}
    for name, rel in MODELS:
        rows = collect(os.path.join(FE, rel))
        if rows:
            data[name] = {(x['run'], x['split']): x for x in rows}
    names = [n for n, _ in MODELS if n in data]
    required = [n for n in [BASE, 'CADE-mini full', 'CADE-mini encoder-only', 'Quantum'] if n in data]
    common = sorted(set.intersection(*[set(data[n]) for n in required])) if required else []
    lines = []
    def log(s=''):
        print(s, flush=True); lines.append(s)
    log('=' * 118); log('FULL NEW EXPERIMENTS -- PCA-5, paired by (run, split)'); log('=' * 118)
    for n in names:
        p = next(iter(data[n].values()))['n_params']
        log(f'  {n:<22} tasks {len(data[n]):>2}   params {p}')
    log(f'  Paired tasks (present for the 4 main models): {len(common)}')
    if not common:
        return
    fams = []
    for k in common:
        f = data[required[0]][k]['family']
        if f not in fams:
            fams.append(f)

    def vals(n, m, keys=common):
        return np.array([data[n][k].get(m, np.nan) if k in data[n] else np.nan for k in keys], float)

    do_cal = not args.no_calibration
    if do_cal:
        log(f'\nCalibrating thresholds (target {args.target_fpr:.0%} of known TRAINING samples flagged) ...')
        calibrate(data, names, common, args.data_dir, args.prefix, args.target_fpr, log)

    log('\nOVERALL (mean +- population std over paired tasks)')
    log(f"  {'metric':<24}" + ''.join(f'{n:>25}' for n in names))
    for m in MAIN + [f'Flagged@T{T}' for T in THRESHOLDS[1:]]:
        log(f'  {LABEL.get(m, m):<24}' + ''.join(f'{np.nanmean(vals(n, m)):>17.3f}+-{np.nanstd(vals(n, m)):<6.3f}' for n in names))
    log(f'  Threshold-free F1 flag-all floor = {FLAG_ALL_F1}.')

    if do_cal:
        log(f'\nTHRESHOLD CALIBRATION  (is 3.5 equally strict for every model? matched false-alarm budget '
            f'= {args.target_fpr:.0%} of known TRAINING samples; the OOD family is never used)')
        log(f"  {'metric':<28}" + ''.join(f'{n:>25}' for n in names))
        for m in ['FPR_known@3.5', 'TrainExceed@3.5', 'T_cal', 'FPR_known@Tcal', 'F1fixed@Tcal', 'CADE_F1@Tcal']:
            log(f'  {LABEL.get(m, m):<28}' + ''.join(f'{np.nanmean(vals(n, m)):>17.3f}+-{np.nanstd(vals(n, m)):<6.3f}' for n in names))
        log('  Read: similar "known-test FPR at 3.5" across models -> 3.5 is equally strict for all.'
            '  "known-test FPR at T_cal" above the target = the model is tighter on training than on unseen known data.')

    log(f'\nPAIRED TESTS  (A minus B; two-sided Wilcoxon signed-rank; n = {len(common)})')
    for A, B in TESTS:
        if A not in data or B not in data:
            continue
        log(f'\n  {A}  vs  {B}')
        log(f"    {'metric':<24}{'mean diff':>11}{'A wins':>8}{'B wins':>8}{'ties':>6}{'p':>9}")
        for m in MAIN + (['CADE_F1@Tcal', 'F1fixed@Tcal'] if do_cal else []):
            d = vals(A, m) - vals(B, m); d = d[~np.isnan(d)]
            if len(d) == 0:
                continue
            wa, wb = int((d > 1e-12).sum()), int((d < -1e-12).sum())
            try:
                p = wilcoxon(d).pvalue if np.any(np.abs(d) > 1e-12) else 1.0
            except ValueError:
                p = np.nan
            log(f'    {LABEL.get(m, m):<24}{d.mean():>+11.3f}{wa:>8}{wb:>8}{len(d) - wa - wb:>6}{p:>9.4f}')
    log('\n  n = 12 -> smallest possible two-sided Wilcoxon p ~ 0.0005; n = 4 (one seed) -> 0.125.')

    for m in ['ROC_raw', 'ROC_mad', 'TF_F1', 'MAD_F1_fixed', 'F1@T3.5'] + (['FPR_known@3.5', 'T_cal', 'CADE_F1@Tcal'] if do_cal else []):
        log(f'\nPER FAMILY: {LABEL.get(m, m)} (mean +- std over runs)')
        log(f"  {'family':<14}" + ''.join(f'{n:>25}' for n in names))
        for fam in fams:
            ks = [k for k in common if data[required[0]][k]['family'] == fam]
            log(f'  {fam:<14}' + ''.join(f'{np.nanmean(vals(n, m, ks)):>17.3f}+-{np.nanstd(vals(n, m, ks)):<6.3f}' for n in names))

    os.makedirs(OUTD, exist_ok=True)
    open(os.path.join(OUTD, 'comparison_summary.txt'), 'w').write('\n'.join(lines) + '\n')
    allm = ['family', 'n_params', 'best_epoch'] + [k for k, _ in HEADLINE] + (CAL if do_cal else [])
    with open(os.path.join(OUTD, 'comparison_per_task.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['model', 'run', 'split'] + allm)
        for n in names:
            for k in common:
                if k in data[n]:
                    w.writerow([n, k[0], k[1]] + [data[n][k].get(c, '') for c in allm])

    # ------------------------------------------------------------------ plots
    pdir = os.path.join(OUTD, 'plots'); os.makedirs(pdir, exist_ok=True)
    cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # 1. headline bars
    head = ['ROC_raw', 'ROC_mad', 'PR_raw', 'TF_F1', 'MAD_F1_fixed', 'F1@T3.5']
    fig, a = plt.subplots(figsize=(12, 4.8)); w = 0.8 / len(names); x0 = np.arange(len(head))
    for i, n in enumerate(names):
        a.bar(x0 + (i - (len(names) - 1) / 2) * w, [np.nanmean(vals(n, m)) for m in head], w,
              yerr=[np.nanstd(vals(n, m)) for m in head], capsize=2, label=n, color=cols[i % len(cols)])
    a.set_xticks(x0); a.set_xticklabels([LABEL.get(m, m) for m in head], fontsize=8); a.set_ylim(0, 1.05)
    a.axhline(FLAG_ALL_F1, color='k', ls=':', lw=1); a.grid(alpha=0.3, axis='y'); a.legend(fontsize=7, ncol=3)
    a.set_title('PCA-5: all models (mean +- std over paired tasks; dotted = flag-all F1)')
    fig.tight_layout(); fig.savefig(os.path.join(pdir, 'headline_metrics.png'), dpi=140); plt.close(fig)
    # 2. threshold sweep
    fig, a = plt.subplots(figsize=(7, 4.5))
    for i, n in enumerate(names):
        a.errorbar(THRESHOLDS, [np.nanmean(vals(n, f'F1@T{T}')) for T in THRESHOLDS],
                   yerr=[np.nanstd(vals(n, f'F1@T{T}')) for T in THRESHOLDS], marker='o', capsize=3,
                   label=n, color=cols[i % len(cols)])
    a.axhline(FLAG_ALL_F1, color='k', ls=':', lw=1)
    a.set_xlabel('T_MAD'); a.set_ylabel('best F1 (CADE protocol)'); a.set_title('Threshold sweep')
    a.grid(alpha=0.3); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(pdir, 'threshold_sweep.png'), dpi=140); plt.close(fig)
    # 3. per-family bars for the key metrics
    for m in ['ROC_raw', 'ROC_mad', 'TF_F1', 'MAD_F1_fixed']:
        fig, a = plt.subplots(figsize=(10, 4.5)); x0 = np.arange(len(fams))
        for i, n in enumerate(names):
            mm, ss = [], []
            for fam in fams:
                v = vals(n, m, [k for k in common if data[required[0]][k]['family'] == fam])
                mm.append(np.nanmean(v)); ss.append(np.nanstd(v))
            a.bar(x0 + (i - (len(names) - 1) / 2) * w, mm, w, yerr=ss, capsize=2, label=n, color=cols[i % len(cols)])
        a.set_xticks(x0); a.set_xticklabels(fams); a.set_ylim(0, 1.05); a.grid(alpha=0.3, axis='y')
        a.legend(fontsize=7, ncol=3); a.set_title(f'Per family: {LABEL.get(m, m)}')
        fig.tight_layout(); fig.savefig(os.path.join(pdir, f'per_family_{m}.png'), dpi=140); plt.close(fig)
    # 4. paired differences: Quantum minus each classical model, per task
    if 'Quantum' in data:
        opp = [n for n in ['CADE-mini full', 'CADE-mini encoder-only', BASE] if n in data]
        fig, ax = plt.subplots(1, len(opp), figsize=(5 * len(opp), 4.5), sharey=True)
        ax = np.atleast_1d(ax)
        for a, B in zip(ax, opp):
            D = [vals('Quantum', m) - vals(B, m) for m in head]
            a.boxplot(D, showmeans=True)
            a.set_xticks(range(1, len(head) + 1)); a.set_xticklabels([LABEL.get(m, m) for m in head])
            for j, d in enumerate(D):
                a.scatter(np.full(len(d), j + 1) + np.random.uniform(-0.12, 0.12, len(d)), d, s=10, alpha=0.6)
            a.axhline(0, color='k', lw=1); a.set_title(f'Quantum minus {B} (per task)', fontsize=9)
            a.tick_params(axis='x', labelsize=7, rotation=30); a.grid(alpha=0.3, axis='y')
        fig.tight_layout(); fig.savefig(os.path.join(pdir, 'paired_differences.png'), dpi=140); plt.close(fig)
    # 6. structure-aware statistics
    if not args.no_stats:
        fam_of = {}
        for n in names:
            for k, row in data[n].items():
                fam_of[k] = row['family']
        pairs = KEY_PAIRS if args.stats_pairs == 'key' else TESTS
        log('\n' + '=' * 118)
        log(f'STRUCTURE-AWARE STATISTICS  (n_boot = {args.n_boot})')
        log('  DeLong: per task, same test samples, p < 0.05. Per-family string: one character per seed (A = A sig. better,'
            ' B = B sig. better, - = n.s.).')
        log('  Boot: per task, paired bootstrap over test samples, 95% CI of the difference.')
        log('  Hierarchical bootstrap: families -> seeds -> test samples resampled; the CI accounts for all three levels.')
        log('=' * 118)
        run_statistics(data, names, fam_of, args.n_boot, pairs, log, OUTD)
        open(os.path.join(OUTD, 'comparison_summary.txt'), 'w').write('\n'.join(lines) + '\n')
    # 5. threshold calibration
    if do_cal:
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
        a = ax[0]; x0 = np.arange(len(names))
        a.bar(x0 - 0.2, [np.nanmean(vals(n, 'FPR_known@3.5')) for n in names], 0.4,
              yerr=[np.nanstd(vals(n, 'FPR_known@3.5')) for n in names], capsize=2, label='at T = 3.5')
        a.bar(x0 + 0.2, [np.nanmean(vals(n, 'FPR_known@Tcal')) for n in names], 0.4,
              yerr=[np.nanstd(vals(n, 'FPR_known@Tcal')) for n in names], capsize=2, label='at calibrated T_cal')
        a.axhline(args.target_fpr, color='k', ls=':', lw=1, label=f'target {args.target_fpr:.0%}')
        a.set_xticks(x0); a.set_xticklabels(names, rotation=25, fontsize=8); a.set_ylabel('false-alarm rate on KNOWN test samples')
        a.set_title('Is T = 3.5 equally strict for every model?'); a.grid(alpha=0.3, axis='y'); a.legend(fontsize=8)
        a = ax[1]
        a.bar(x0 - 0.2, [np.nanmean(vals(n, 'F1@T3.5')) for n in names], 0.4,
              yerr=[np.nanstd(vals(n, 'F1@T3.5')) for n in names], capsize=2, label='CADE F1 at T = 3.5')
        a.bar(x0 + 0.2, [np.nanmean(vals(n, 'CADE_F1@Tcal')) for n in names], 0.4,
              yerr=[np.nanstd(vals(n, 'CADE_F1@Tcal')) for n in names], capsize=2, label='CADE F1 at matched false-alarm T_cal')
        a.set_xticks(x0); a.set_xticklabels(names, rotation=25, fontsize=8); a.set_ylim(0, 1.05)
        a.set_title('Detection at a fixed vs matched false-alarm threshold'); a.grid(alpha=0.3, axis='y'); a.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(pdir, 'threshold_calibration.png'), dpi=140); plt.close(fig)
    print(f'\nSaved: {OUTD}/comparison_summary.txt, comparison_per_task.csv, plots/*.png'
          + ('' if args.no_stats else ', delong_per_task.csv, bootstrap_per_task.csv, hierarchical_bootstrap.csv'))


if __name__ == '__main__':
    main()
