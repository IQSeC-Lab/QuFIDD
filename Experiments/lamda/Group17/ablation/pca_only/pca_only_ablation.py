#!/usr/bin/env python3
"""
pca_only_ablation.py  --  Group17 ablation 1: PCA only, NO trainable model
===========================================================================
Question: how much of Group17's OOD ranking comes from the frozen PCA front
end alone, with no VQC and no training at all?

For every (run, split) task of Group17 this script:
  1. Loads the EXACT PCA that Group17 fitted for that task
     (../../results/run{r}/split{s}/pca_components.pkl), so the comparison
     with the trained model is paired. If the pickle is missing it refits
     PCA(32) on X_train and says so.
  2. Projects X_train / X_test to 32 dims and L2-normalizes -- identical to
     Group17's PCACompressor.transform.
  3. Scores test samples against the centroids of the KNOWN classes only,
     two ways:
       euclidean (primary): d_c = || z - mu_c ||_2
       cosine:              d_c = 1 - cos(z, mu_c)
     raw score = min_c d_c ; MAD score = min_c |d_c - median_c| / MAD_c
     (per-class median/MAD of training distances, b = 1.4826, floor 1e-6).
  4. Evaluates with Group17's OWN evaluate_ood / evaluate_mad_threshold
     (imported from the Group17 script), so every metric is computed exactly
     as for the trained model.
  5. Writes Group17-format results so cade_protocol_eval.py can also be run
     on them, and an aggregate that compares against trained Group17.

Layout:
  Group17/
    Group17_vqc_fixed_target_fidelity_4class.py
    results/run*/split*/{summary.json, ood_scores.csv, pca_components.pkl}
    ablation/pca_only/
      pca_only_ablation.py                         <- this script
      results/run*/split*/{summary.json, ood_scores.csv}          (euclidean)
      results_cosine/run*/split*/{summary.json, ood_scores.csv}   (cosine)
      aggregate.csv, aggregate_summary.txt

Usage (from Group17/ablation/pca_only/, OOD_env active; runs in minutes):
  python3 pca_only_ablation.py
"""

import os, sys, csv, json, pickle, argparse, importlib.util
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score

HERE      = os.path.dirname(os.path.abspath(__file__))
GROUP_DIR = os.path.dirname(os.path.dirname(HERE))
G17_FILE  = os.path.join(GROUP_DIR, 'Group17_vqc_fixed_target_fidelity_4class.py')
SEEDS     = {0: 42, 1: 123, 2: 456}
PCA_DIM   = 32


def load_g17():
    spec = importlib.util.spec_from_file_location('g17', G17_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_split(data_dir, prefix, s):
    d = np.load(os.path.join(data_dir, f'{prefix}_{s}.npz'), allow_pickle=True)
    def dense(a):
        return a.item().toarray().astype(np.float32) if a.shape == () else a.astype(np.float32)
    hf = d['heldout_family']
    fam = str(hf.item()) if hf.shape == () else str(hf)
    return (dense(d['X_train']), d['y_train'].astype(np.int64),
            dense(d['X_test']),  d['y_test'].astype(np.int64), fam)


def pca_transform(pca, X):
    z = pca.transform(X).astype(np.float32)
    return z / np.linalg.norm(z, axis=1, keepdims=True).clip(min=1e-8)


def centroid_scores(z_tr, y_tr, z_te, metric):
    classes = sorted(int(c) for c in np.unique(y_tr))
    cents = {c: z_tr[y_tr == c].mean(axis=0) for c in classes}

    def dist(Z, c):
        if metric == 'euclidean':
            return np.linalg.norm(Z - cents[c], axis=1)
        mu = cents[c] / (np.linalg.norm(cents[c]) + 1e-8)
        return 1.0 - Z @ mu                      # Z rows are unit-norm

    stats = {}
    for c in classes:
        dc = dist(z_tr[y_tr == c], c)
        med = np.median(dc)
        stats[c] = (med, max(1.4826 * np.median(np.abs(dc - med)), 1e-6))
    D = np.stack([dist(z_te, c) for c in classes], axis=1)
    A = np.stack([np.abs(D[:, j] - stats[c][0]) / stats[c][1]
                  for j, c in enumerate(classes)], axis=1)
    return D.min(axis=1), A.min(axis=1)


def write_task(out_root, r, s, fam, seed, n_tr, n_te, n_feat, pca_src, metric,
               mad_res, tf_res, raw, mad, extra):
    d = os.path.join(out_root, f'run{r}', f'split{s}')
    os.makedirs(d, exist_ok=True)
    summary = {
        'model': f'Group17_ablation_PCAonly_{metric}',
        'description': f'Frozen PCA(32)+L2 norm only, no VQC, no training; '
                       f'{metric} distance to known-class centroids',
        'run_id': r, 'split_id': s, 'seed': seed, 'ood_family': fam,
        'n_train': n_tr, 'n_test': n_te, 'n_features': n_feat,
        'pca_source': pca_src, 'n_trainable_params': 0,
        'MAD_Threshold': mad_res, 'Threshold_Free': tf_res,
    }
    summary.update(extra)
    with open(os.path.join(d, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(d, 'ood_scores.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['raw_score', 'mad_score'])
        for a, b in zip(raw, mad):
            w.writerow([a, b])


def trained_reference(r, s, y_test, ood_label):
    """Trained Group17 metrics for the same task (read from its saved files)."""
    d = os.path.join(GROUP_DIR, 'results', f'run{r}', f'split{s}')
    try:
        sm = json.load(open(os.path.join(d, 'summary.json')))
        rows = list(csv.DictReader(open(os.path.join(d, 'ood_scores.csv'))))
        mad = np.nan_to_num(np.array([float(x['mad_score']) for x in rows]), nan=0.0, posinf=1e12)
        ood = (y_test == ood_label).astype(int)
        return {'ROC_raw': sm['Threshold_Free']['roc_auc'], 'PR_raw': sm['Threshold_Free']['pr_auc'],
                'TF_F1': sm['Threshold_Free']['best_f1'], 'MAD_F1': sm['MAD_Threshold']['best_f1'],
                'ROC_mad': float(roc_auc_score(ood, mad))}
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='~/projects/OOD/Data/cade_lamda_4class_vt')
    ap.add_argument('--prefix', default='drebin_new')
    args = ap.parse_args()
    data_dir = os.path.expanduser(args.data_dir)
    g17 = load_g17()

    lines = []
    def log(s=''):
        print(s, flush=True); lines.append(s)

    log('=' * 88)
    log('Group17 ABLATION 1 -- PCA only (no VQC, no training), scored against known-class centroids')
    log(f'data: {data_dir}')
    log('=' * 88)

    n_splits = 0
    while os.path.isdir(os.path.join(GROUP_DIR, 'results', 'run0', f'split{n_splits}')):
        n_splits += 1
    if n_splits == 0:
        n_splits = g17.N_SPLITS
        log(f'No Group17 results found; using N_SPLITS={n_splits} and refitting PCA.')

    rows = []
    for r in range(3):
        for s in range(n_splits):
            X_tr, y_tr, X_te, y_te, fam = load_split(data_dir, args.prefix, s)
            ood_label = (set(np.unique(y_te).tolist()) - set(np.unique(y_tr).tolist())).pop()
            pkl = os.path.join(GROUP_DIR, 'results', f'run{r}', f'split{s}', 'pca_components.pkl')
            if os.path.exists(pkl):
                pca = pickle.load(open(pkl, 'rb')); pca_src = 'Group17 saved PCA (paired)'
            else:
                np.random.seed(SEEDS[r])
                pca = PCA(n_components=PCA_DIM).fit(X_tr); pca_src = 'refit (Group17 pickle missing)'
                log(f'  run{r}/split{s}: pca_components.pkl missing -> refit')
            z_tr, z_te = pca_transform(pca, X_tr), pca_transform(pca, X_te)
            ev = float(np.sum(pca.explained_variance_ratio_))
            ref = trained_reference(r, s, y_te, ood_label)

            for metric, out_name in [('euclidean', 'results'), ('cosine', 'results_cosine')]:
                raw, mad = centroid_scores(z_tr, y_tr, z_te, metric)
                tf_res  = g17.evaluate_ood(raw, y_te, ood_label)
                mad_res = g17.evaluate_mad_threshold(mad, y_te, ood_label)
                ood = (y_te == ood_label).astype(int)
                roc_mad = float(roc_auc_score(ood, mad))
                write_task(os.path.join(HERE, out_name), r, s, fam, SEEDS[r], len(y_tr), len(y_te),
                           X_tr.shape[1], pca_src, metric, mad_res, tf_res, raw, mad,
                           {'pca_explained_variance': ev, 'roc_auc_mad_score': roc_mad})
                rows.append({'variant': f'PCA-only ({metric})', 'run_id': r, 'split_id': s,
                             'ood_family': fam, 'ROC_raw': tf_res['roc_auc'], 'ROC_mad': roc_mad,
                             'PR_raw': tf_res['pr_auc'], 'TF_F1': tf_res['best_f1'],
                             'MAD_F1': mad_res['best_f1']})
            if ref is not None:
                rows.append({'variant': 'Trained Group17', 'run_id': r, 'split_id': s,
                             'ood_family': fam, **ref})
            log(f'  run{r}/split{s} {fam:<12} PCA var={ev:.3f}  '
                f'ROC eucl={rows[-3 if ref else -2]["ROC_raw"]:.3f}  '
                f'cos={rows[-2 if ref else -1]["ROC_raw"]:.3f}'
                + (f'  | trained G17={ref["ROC_raw"]:.3f}' if ref else ''))
    summarize(rows, log)

    keys = ['variant', 'run_id', 'split_id', 'ood_family', 'ROC_raw', 'ROC_mad', 'PR_raw', 'TF_F1', 'MAD_F1']
    with open(os.path.join(HERE, 'aggregate.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for x in rows:
            w.writerow({k: x.get(k) for k in keys})
    with open(os.path.join(HERE, 'aggregate_summary.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nSaved: {HERE}/results*/run*/split*/, aggregate.csv, aggregate_summary.txt')


def summarize(rows, log):
    variants = []
    for x in rows:
        if x['variant'] not in variants:
            variants.append(x['variant'])
    fams = []
    for x in rows:
        if x['ood_family'] not in fams:
            fams.append(x['ood_family'])
    mets = ['ROC_raw', 'ROC_mad', 'PR_raw', 'TF_F1', 'MAD_F1']
    log('')
    log('Mean +- std over runs (population std). ROC_raw/PR_raw/TF_F1 rank by the raw distance;')
    log('ROC_mad ranks by the MAD score; MAD_F1 = fixed threshold 3.5. Flag-all F1 floor = 0.769.')
    for fam in fams + ['ALL']:
        log(f'\n  {fam}')
        log(f"    {'variant':<26}" + ''.join(f'{m:>15}' for m in mets))
        for v in variants:
            sel = [x for x in rows if x['variant'] == v and (fam == 'ALL' or x['ood_family'] == fam)]
            if not sel:
                continue
            line = f'    {v:<26}'
            for m in mets:
                a = np.array([x[m] for x in sel], float)
                line += f'{a.mean():>8.3f}+-{a.std():<5.3f}'
            log(line)


if __name__ == '__main__':
    main()
