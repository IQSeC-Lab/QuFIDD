#!/usr/bin/env python3
"""
pca5_only_baseline.py  --  zero-parameter PCA-5 baseline for Groups 18/20
=========================================================================
No model, no training. For every (run, split) task that Group20 has run, this
uses the EXACT PCA-5 Group20 fitted for that task (pca_components.pkl), so the
input is identical to Group20's and Group18's, and scores test samples by
Euclidean distance to the known-class centroids of the L2-normalized PCA-5
vectors (the same baseline Group18/20 report as 'PCA-5 centroid ROC').

Reports, per task and aggregated:
  MAD F1      fixed threshold T = 3.5 on A = min_c |d_c - median_c| / MAD_c
  TF F1       threshold-free: rank by raw distance, best F1 over all cutoffs
              (flag-all floor on these test sets = 0.769)
  CADE F1     CADE's Table-3 metric: flag A > 3.5, rank flagged by distance, best F1
  ROC / PR    on the raw distance; ROC also on the MAD score

Layout:
  Group20/
    results_matched/run*/split*/pca_components.pkl
    ablation/pca5_only/
      pca5_only_baseline.py            <- this script
      run*/split*/summary.json, aggregate_summary.txt

Usage (from Group20/ablation/pca5_only/; seconds, no torch needed):
  python3 pca5_only_baseline.py
  python3 pca5_only_baseline.py --results-dir ../../results      # if Group20 used the other folder
"""

import os, json, pickle, argparse
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score

HERE  = os.path.dirname(os.path.abspath(__file__))
SEEDS = {0: 42, 1: 123, 2: 456}


def dense(a):
    return a.item().toarray().astype(np.float32) if a.shape == () else a.astype(np.float32)


def l2n(Z):
    return Z / np.linalg.norm(Z, axis=1, keepdims=True).clip(min=1e-8)


def best_f1(scores, ood, idx=None):
    idx = np.arange(len(scores)) if idx is None else idx
    if len(idx) == 0:
        return 0.0, 0.0, 0.0
    order = idx[np.argsort(-scores[idx], kind='stable')]
    tp = np.cumsum(ood[order]); k = np.arange(1, len(order) + 1)
    p, r = tp / k, tp / ood.sum()
    f = np.where(p + r > 0, 2 * p * r / (p + r + 1e-12), 0.0); b = int(np.argmax(f))
    return float(f[b]), float(p[b]), float(r[b])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', default=os.path.join(HERE, '..', '..', 'results_matched'),
                    help="Group20 results folder holding each task's pca_components.pkl")
    ap.add_argument('--data-dir', default='~/projects/OOD/Data/cade_lamda_4class_vt')
    ap.add_argument('--prefix', default='drebin_new')
    ap.add_argument('--pca-dim', type=int, default=5, help='used only if a pca pickle is missing (refit)')
    a = ap.parse_args()
    res_dir, data_dir = os.path.abspath(os.path.expanduser(a.results_dir)), os.path.expanduser(a.data_dir)

    lines = []
    def log(s=''):
        print(s, flush=True); lines.append(s)
    log('=' * 96)
    log('PCA-5 ONLY (0 parameters) -- same PCA as Group20, L2-normalized, distance to known-class centroids')
    log(f'PCA source: {res_dir}')
    log('=' * 96)

    rows, cache = [], {}
    for r in range(3):
        for s in range(4):
            tdir = os.path.join(res_dir, f'run{r}', f'split{s}')
            if not os.path.isdir(tdir):
                continue
            if s not in cache:
                z = np.load(os.path.join(data_dir, f'{a.prefix}_{s}.npz'), allow_pickle=True)
                hf = z['heldout_family']
                cache[s] = (dense(z['X_train']), z['y_train'], dense(z['X_test']), z['y_test'],
                            str(hf.item()) if hf.shape == () else str(hf))
            Xtr, ytr, Xte, yte, fam = cache[s]
            pkl = os.path.join(tdir, 'pca_components.pkl')
            if os.path.exists(pkl):
                pca, src = pickle.load(open(pkl, 'rb')), 'Group20 saved PCA'
            else:
                pca, src = PCA(n_components=a.pca_dim, svd_solver='full').fit(Xtr), 'refit'
            Ztr, Zte = l2n(pca.transform(Xtr)), l2n(pca.transform(Xte))
            ood = (~np.isin(yte, np.unique(ytr))).astype(int)

            D, A = [], []
            for c in np.unique(ytr):
                mu = Ztr[ytr == c].mean(axis=0)
                d = np.linalg.norm(Ztr[ytr == c] - mu, axis=1)
                med = np.median(d); mad = max(1.4826 * np.median(np.abs(d - med)), 1e-12)
                dt = np.linalg.norm(Zte - mu, axis=1)
                D.append(dt); A.append(np.abs(dt - med) / mad)
            raw, mad_sc = np.min(D, axis=0), np.min(A, axis=0)

            pred = mad_sc > 3.5
            tp = int((pred & (ood == 1)).sum()); fp = int((pred & (ood == 0)).sum()); fn = int(((~pred) & (ood == 1)).sum())
            P = tp / (tp + fp) if tp + fp else 0.0; R = tp / (tp + fn) if tp + fn else 0.0
            mad_f1 = 2 * P * R / (P + R) if P + R else 0.0
            tf, tfp, tfr = best_f1(raw, ood)
            cade, _, _ = best_f1(raw, ood, np.where(pred)[0])
            res = {'run_id': r, 'split_id': s, 'seed': SEEDS[r], 'ood_family': fam, 'pca_source': src,
                   'n_trainable_params': 0,
                   'MAD_F1_T3.5': mad_f1, 'MAD_precision': P, 'MAD_recall': R, 'MAD_frac_flagged': float(pred.mean()),
                   'TF_F1': tf, 'TF_precision': tfp, 'TF_recall': tfr, 'CADE_F1_T3.5': cade,
                   'ROC_raw': float(roc_auc_score(ood, raw)), 'ROC_mad': float(roc_auc_score(ood, mad_sc)),
                   'PR_raw': float(average_precision_score(ood, raw))}
            od = os.path.join(HERE, f'run{r}', f'split{s}'); os.makedirs(od, exist_ok=True)
            json.dump(res, open(os.path.join(od, 'summary.json'), 'w'), indent=2)
            rows.append(res)
            log(f'  run{r}/split{s} {fam:<12} MAD_F1={mad_f1:.3f} (flagged {pred.mean():.3f})  TF_F1={tf:.3f}  '
                f'CADE_F1={cade:.3f}  ROC_raw={res["ROC_raw"]:.3f}  ROC_mad={res["ROC_mad"]:.3f}')

    if not rows:
        log('No Group20 task folders found -- check --results-dir.'); return
    keys = ['MAD_F1_T3.5', 'TF_F1', 'CADE_F1_T3.5', 'ROC_raw', 'ROC_mad', 'PR_raw', 'MAD_frac_flagged']
    log('')
    log(f"  {'family':<14}{'n':>3}" + ''.join(f'{k:>16}' for k in keys))
    fams = []
    for x in rows:
        if x['ood_family'] not in fams:
            fams.append(x['ood_family'])
    for fam in fams + ['ALL']:
        sel = rows if fam == 'ALL' else [x for x in rows if x['ood_family'] == fam]
        M = np.array([[x[k] for k in keys] for x in sel], float)
        log(f'  {fam:<14}{len(sel):>3}' + ''.join(f'{m:>9.3f}+-{sd:<5.3f}' for m, sd in zip(M.mean(0), M.std(0))))
    log('  TF_F1 flag-all floor = 0.769. std = population std over runs.')
    open(os.path.join(HERE, 'aggregate_summary.txt'), 'w').write('\n'.join(lines) + '\n')
    print(f'\nSaved: {HERE}/run*/split*/summary.json, aggregate_summary.txt')


if __name__ == '__main__':
    main()
