#!/usr/bin/env python3
"""
pca_ensemble_eval.py  --  Does the learned score add anything to PCA?
======================================================================
No retraining. For every (run, split) task of the group this script sits in:

  1. Builds the PCA-only score from the group's OWN saved PCA
     (../../results/run{r}/split{s}/pca_components.pkl; refits PCA(32) if
     missing): Euclidean distance of the L2-normalized PCA vector to the
     nearest known-class centroid. This is the zero-parameter baseline.
  2. Reads the model's saved scores (../../results/run{r}/split{s}/ood_scores.csv):
     the raw score (distance to nearest class) and the MAD score.
  3. Combines each model score with the PCA score by RANK AVERAGING
     (scores on different scales are made comparable by converting each to
     ranks in [0, 1] over the test set, then averaging).
       PRIMARY: equal weights, w = 0.5 (fixed in advance, not tuned).
       Sensitivity only: w in {0, 0.25, 0.5, 0.75, 1}, where
       w = weight on the model score (w=0 -> PCA only, w=1 -> model only).
       Picking the best w per result would use test labels -- do not report
       the best w as a result.
  4. Reports ROC-AUC, PR-AUC and threshold-free best F1 for each variant,
     plus the Spearman rank correlation between the model score and the
     PCA score (1.0 = the model score carries no information PCA lacks).

Reading the result: if the w=0.5 ensemble beats PCA-only, the learned score
is COMPLEMENTARY to PCA (training learned something PCA does not have). If
it does not, the learned score is a degraded copy of PCA's information.

Layout (copy this script into each group):
  GroupXX/
    results/run*/split*/{ood_scores.csv, pca_components.pkl}
    analysis/pca_ensemble/
      pca_ensemble_eval.py                    <- this script
      run0/split0/pca_ensemble.json           <- per task
      run0_results.csv ...                    <- per run
      aggregate.csv, aggregate_summary.txt

Usage (from GroupXX/analysis/pca_ensemble/, OOD_env active; runs in minutes):
  python3 pca_ensemble_eval.py
"""

import os, sys, csv, json, pickle, argparse
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import rankdata, spearmanr

HERE       = os.path.dirname(os.path.abspath(__file__))
GROUP_DIR  = os.path.dirname(os.path.dirname(HERE))
GROUP_NAME = os.path.basename(GROUP_DIR)
SEEDS      = {0: 42, 1: 123, 2: 456}
PCA_DIM    = 32
WEIGHTS    = [0.0, 0.25, 0.5, 0.75, 1.0]
PRIMARY_W  = 0.5


def load_split(data_dir, prefix, s):
    d = np.load(os.path.join(data_dir, f'{prefix}_{s}.npz'), allow_pickle=True)
    def dense(a):
        return a.item().toarray().astype(np.float32) if a.shape == () else a.astype(np.float32)
    hf = d['heldout_family']
    fam = str(hf.item()) if hf.shape == () else str(hf)
    return dense(d['X_train']), d['y_train'].astype(np.int64), dense(d['X_test']), d['y_test'].astype(np.int64), fam


def pca_only_score(pca, X_tr, y_tr, X_te):
    def tf(X):
        z = pca.transform(X).astype(np.float32)
        return z / np.linalg.norm(z, axis=1, keepdims=True).clip(min=1e-8)
    z_tr, z_te = tf(X_tr), tf(X_te)
    classes = np.unique(y_tr)
    D = np.stack([np.linalg.norm(z_te - z_tr[y_tr == c].mean(axis=0), axis=1) for c in classes], axis=1)
    return D.min(axis=1)


def to_rank(x):
    return (rankdata(x) - 1) / max(len(x) - 1, 1)


def best_f1(scores, is_ood):
    order = np.argsort(-scores, kind='stable')
    tp = np.cumsum(is_ood[order]); k = np.arange(1, len(order) + 1)
    p, r = tp / k, tp / is_ood.sum()
    return float(np.max(np.where(p + r > 0, 2 * p * r / (p + r + 1e-12), 0.0)))


def metrics(scores, is_ood):
    return {'ROC_AUC': float(roc_auc_score(is_ood, scores)),
            'PR_AUC':  float(average_precision_score(is_ood, scores)),
            'F1':      best_f1(scores, is_ood)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='~/projects/OOD/Data/cade_lamda_4class_vt')
    ap.add_argument('--prefix', default='drebin_new')
    args = ap.parse_args()
    data_dir = os.path.expanduser(args.data_dir)
    res_dir  = os.path.join(GROUP_DIR, 'results')

    lines = []
    def log(s=''):
        print(s, flush=True); lines.append(s)
    log('=' * 92)
    log(f'{GROUP_NAME} -- PCA-only + model score ensemble (rank average). Primary: w = {PRIMARY_W} '
        f'(fixed in advance)')
    log(f'results: {res_dir}')
    log('=' * 92)

    n_splits = 0
    while os.path.isdir(os.path.join(res_dir, 'run0', f'split{n_splits}')):
        n_splits += 1
    if n_splits == 0:
        sys.exit(f'No results under {res_dir}')

    rows, missing, cache = [], [], {}
    for r in range(3):
        run_rows = []
        for s in range(n_splits):
            tdir = os.path.join(res_dir, f'run{r}', f'split{s}')
            sc_path = os.path.join(tdir, 'ood_scores.csv')
            if not os.path.exists(sc_path):
                missing.append((r, s)); continue
            if s not in cache:
                cache[s] = load_split(data_dir, args.prefix, s)
            X_tr, y_tr, X_te, y_te, fam = cache[s]
            is_ood = (~np.isin(y_te, np.unique(y_tr))).astype(int)

            pkl = os.path.join(tdir, 'pca_components.pkl')
            if os.path.exists(pkl):
                pca = pickle.load(open(pkl, 'rb')); pca_src = 'saved'
            else:
                pca = PCA(n_components=PCA_DIM, svd_solver='full').fit(X_tr); pca_src = 'refit'
            s_pca = pca_only_score(pca, X_tr, y_tr, X_te)

            sc = list(csv.DictReader(open(sc_path)))
            rk = 'raw_score' if 'raw_score' in sc[0] else 'raw_euclidean_score'
            s_raw = np.array([float(x[rk]) for x in sc])
            s_mad = np.nan_to_num(np.array([float(x['mad_score']) for x in sc]), nan=0.0, posinf=1e12)
            if len(s_raw) != len(y_te):
                sys.exit(f'Length mismatch: {sc_path}')

            task = {'group': GROUP_NAME, 'run_id': r, 'split_id': s, 'seed': SEEDS.get(r),
                    'ood_family': fam, 'pca_source': pca_src, 'primary_weight': PRIMARY_W,
                    'spearman_raw_vs_pca': float(spearmanr(s_raw, s_pca)[0]),
                    'spearman_mad_vs_pca': float(spearmanr(s_mad, s_pca)[0]),
                    'variants': {}}
            r_pca = to_rank(s_pca)
            for name, s_model in [('raw', s_raw), ('mad', s_mad)]:
                r_mod = to_rank(s_model)
                for w in WEIGHTS:
                    m = metrics(w * r_mod + (1 - w) * r_pca, is_ood)
                    task['variants'][f'{name}_w{w}'] = m
                    row = {'run_id': r, 'split_id': s, 'seed': SEEDS.get(r), 'ood_family': fam,
                           'model_score': name, 'w_model': w,
                           'ROC_AUC': round(m['ROC_AUC'], 6), 'PR_AUC': round(m['PR_AUC'], 6),
                           'F1': round(m['F1'], 6),
                           'spearman_vs_pca': round(task[f'spearman_{name}_vs_pca'], 6)}
                    run_rows.append(row)
            d = os.path.join(HERE, f'run{r}', f'split{s}')
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, 'pca_ensemble.json'), 'w') as f:
                json.dump(task, f, indent=2)
            v = task['variants']
            log(f'  run{r}/split{s} {fam:<12} ROC  PCA={v["raw_w0.0"]["ROC_AUC"]:.3f}  '
                f'model_raw={v["raw_w1.0"]["ROC_AUC"]:.3f}  ens_raw={v["raw_w0.5"]["ROC_AUC"]:.3f}  '
                f'model_mad={v["mad_w1.0"]["ROC_AUC"]:.3f}  ens_mad={v["mad_w0.5"]["ROC_AUC"]:.3f}  '
                f'rho(raw,PCA)={task["spearman_raw_vs_pca"]:.2f}')
        if run_rows:
            with open(os.path.join(HERE, f'run{r}_results.csv'), 'w', newline='') as f:
                w_ = csv.DictWriter(f, fieldnames=list(run_rows[0].keys())); w_.writeheader(); w_.writerows(run_rows)
        rows += run_rows

    log(f'\nTasks: {len({(x["run_id"], x["split_id"]) for x in rows})}/{3 * n_splits}   '
        f'Missing: {missing if missing else "none"}')
    fams = []
    for x in rows:
        if x['ood_family'] not in fams:
            fams.append(x['ood_family'])

    agg = []
    def stat(sel, k):
        a = np.array([x[k] for x in sel], float); return a.mean(), a.std()
    label = {0.0: 'PCA only', 0.25: 'w=0.25', 0.5: 'ENSEMBLE w=0.5', 0.75: 'w=0.75', 1.0: 'model only'}
    for name in ['raw', 'mad']:
        log(f'\n--- model score = {name.upper()}  (mean +- std over runs; population std; '
            f'F1 floor = flag-all) ---')
        for fam in fams + ['ALL']:
            sel_f = [x for x in rows if x['model_score'] == name and (fam == 'ALL' or x['ood_family'] == fam)]
            rho = stat([x for x in sel_f if x['w_model'] == 0.0], 'spearman_vs_pca')
            log(f'  {fam}   Spearman(model, PCA) = {rho[0]:.3f}')
            log(f"    {'variant':<16}{'ROC_AUC':>16}{'PR_AUC':>16}{'F1':>16}")
            for w in WEIGHTS:
                sel = [x for x in sel_f if x['w_model'] == w]
                vals = [stat(sel, k) for k in ['ROC_AUC', 'PR_AUC', 'F1']]
                log(f'    {label[w]:<16}' + ''.join(f'{m:>9.3f}+-{sd:<5.3f}' for m, sd in vals))
                rec = {'model_score': name, 'ood_family': fam, 'w_model': w, 'n_tasks': len(sel),
                       'spearman_vs_pca_mean': round(rho[0], 6)}
                for k, (m, sd) in zip(['ROC_AUC', 'PR_AUC', 'F1'], vals):
                    rec[f'{k}_mean'], rec[f'{k}_std'] = round(m, 6), round(sd, 6)
                agg.append(rec)
        allsel = [x for x in rows if x['model_score'] == name]
        pca_roc = stat([x for x in allsel if x['w_model'] == 0.0], 'ROC_AUC')[0]
        ens_roc = stat([x for x in allsel if x['w_model'] == PRIMARY_W], 'ROC_AUC')[0]
        verdict = 'COMPLEMENTARY (ensemble > PCA-only)' if ens_roc > pca_roc + 0.005 else \
                  'NOT complementary (ensemble <= PCA-only)'
        log(f'  => {name.upper()}: ensemble ROC {ens_roc:.3f} vs PCA-only {pca_roc:.3f}  ->  {verdict}')

    with open(os.path.join(HERE, 'aggregate.csv'), 'w', newline='') as f:
        w_ = csv.DictWriter(f, fieldnames=list(agg[0].keys())); w_.writeheader(); w_.writerows(agg)
    with open(os.path.join(HERE, 'aggregate_summary.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nSaved: {HERE}/run*/split*/pca_ensemble.json, run*_results.csv, aggregate.csv, aggregate_summary.txt')


if __name__ == '__main__':
    main()
