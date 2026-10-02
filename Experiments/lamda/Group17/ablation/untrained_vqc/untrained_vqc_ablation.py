#!/usr/bin/env python3
"""
untrained_vqc_ablation.py  --  Group17 ablation 2: the SAME model, NOT trained
==============================================================================
Question: how much of Group17's OOD ranking comes from TRAINING the circuit?

For every (run, split) task this script rebuilds Group17's model exactly as
Group17's run_task does -- VQCEncoder(input_dim, seed=seed), which sets the
seed and draws q_weights ~ N(0, 0.01^2), i.e. the exact near-identity
initial weights that training started from -- and then SKIPS training:
  1. PCA: loads the EXACT PCA Group17 fitted for that task
     (../../results/run{r}/split{s}/pca_components.pkl), so the only
     difference from the trained model is the circuit weights. Refits and
     says so if the pickle is missing.
  2. Simulates the untrained circuit on X_train and X_test (full states).
  3. Scores with Group17's OWN compute_fidelity_target_scores and evaluates
     with its OWN evaluate_ood / evaluate_mad_threshold (all imported from
     the Group17 script), so every number is computed exactly as for the
     trained model.
  4. Writes Group17-format results (summary.json + ood_scores.csv) so
     cade_protocol_eval.py can be run on them, and an aggregate comparing
     untrained vs trained Group17.

No training, but the circuit is still simulated for ~4,000 samples per task,
so use the SLURM array (one task per array index), then run 'aggregate'.

Layout:
  Group17/
    Group17_vqc_fixed_target_fidelity_4class.py
    results/run*/split*/{summary.json, ood_scores.csv, pca_components.pkl}
    ablation/untrained_vqc/
      untrained_vqc_ablation.py                        <- this script
      run_untrained_vqc_ablation.slurm
      results/run*/split*/{summary.json, ood_scores.csv}
      aggregate.csv, aggregate_summary.txt

Usage (from Group17/ablation/untrained_vqc/, OOD_env active):
  python3 untrained_vqc_ablation.py <task_id 0-11>
  python3 untrained_vqc_ablation.py aggregate
"""

import os, sys, csv, json, time, pickle, argparse, importlib.util
import numpy as np
from sklearn.metrics import roc_auc_score

HERE      = os.path.dirname(os.path.abspath(__file__))
GROUP_DIR = os.path.dirname(os.path.dirname(HERE))
G17_FILE  = os.path.join(GROUP_DIR, 'Group17_vqc_fixed_target_fidelity_4class.py')
OUT_ROOT  = os.path.join(HERE, 'results')


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


def run_task(task_id, data_dir, prefix):
    g17 = load_g17()
    r, s = task_id // g17.N_SPLITS, task_id % g17.N_SPLITS
    seed = g17.SEEDS[r]
    X_tr, y_tr, X_te, y_te, fam = load_split(data_dir, prefix, s)
    ood_label = (set(np.unique(y_te).tolist()) - set(np.unique(y_tr).tolist())).pop()
    print('=' * 70, flush=True)
    print(f'Group17 ABLATION 2 -- UNTRAINED VQC | task={task_id} run={r} split={s} '
          f'seed={seed} OOD={fam}', flush=True)
    print('=' * 70, flush=True)

    # Same construction order as Group17.run_task -> identical initial weights
    g17.set_seed(seed)
    model = g17.VQCEncoder(X_tr.shape[1], seed=seed)
    w0 = model.q_weights.detach().cpu().numpy()
    print(f'  q_weights (initial, untrained): shape={w0.shape}  '
          f'mean|w|={np.abs(w0).mean():.5f}  max|w|={np.abs(w0).max():.5f}', flush=True)

    pkl = os.path.join(GROUP_DIR, 'results', f'run{r}', f'split{s}', 'pca_components.pkl')
    if os.path.exists(pkl):
        model.pca_comp.pca = pickle.load(open(pkl, 'rb'))
        model.pca_comp.fitted = True
        pca_src = 'Group17 saved PCA (paired)'
    else:
        model.pca_comp.fit(X_tr)
        pca_src = 'refit (Group17 pickle missing)'
    print(f'  PCA: {pca_src}', flush=True)

    t0 = time.time()
    st_tr = model.get_states_numpy(X_tr)
    st_te = model.get_states_numpy(X_te)
    sim_time = time.time() - t0
    print(f'  Simulated states: train={st_tr.shape} test={st_te.shape} ({sim_time:.1f}s)', flush=True)

    raw, mad = g17.compute_fidelity_target_scores(st_tr, y_tr, st_te, g17.N_CLASSES)
    tf_res  = g17.evaluate_ood(raw, y_te, ood_label)
    mad_res = g17.evaluate_mad_threshold(mad, y_te, ood_label)
    roc_mad = float(roc_auc_score((y_te == ood_label).astype(int), np.nan_to_num(mad, nan=0.0)))

    d = os.path.join(OUT_ROOT, f'run{r}', f'split{s}')
    os.makedirs(d, exist_ok=True)
    summary = {
        'model': 'Group17_ablation_UntrainedVQC',
        'description': 'Group17 model at its initial weights (q_weights ~ N(0,0.01^2), '
                       'same seed/construction as training), no training; fixed-target fidelity scoring',
        'run_id': r, 'split_id': s, 'seed': seed, 'ood_family': fam,
        'n_train': len(y_tr), 'n_test': len(y_te), 'n_features': X_tr.shape[1],
        'pca_source': pca_src, 'trained': False, 'sim_time_sec': round(sim_time, 2),
        'q_weights_mean_abs': float(np.abs(w0).mean()),
        'roc_auc_mad_score': roc_mad,
        'MAD_Threshold': mad_res, 'Threshold_Free': tf_res,
    }
    with open(os.path.join(d, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(d, 'ood_scores.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['raw_score', 'mad_score'])
        for a, b in zip(raw, mad):
            w.writerow([a, b])
    print(f'  [UNTRAINED] TF F1={tf_res["best_f1"]:.4f}  ROC_raw={tf_res["roc_auc"]:.4f}  '
          f'ROC_mad={roc_mad:.4f}  MAD F1={mad_res["best_f1"]:.4f}', flush=True)
    print(f'  Saved: {d}', flush=True)


def aggregate(data_dir, prefix):
    lines = []
    def log(s=''):
        print(s, flush=True); lines.append(s)
    log('=' * 88)
    log('Group17 ABLATION 2 -- UNTRAINED VQC vs TRAINED Group17 (same PCA, same seed, same scoring)')
    log('=' * 88)
    rows, missing, n_splits = [], [], 0
    while os.path.isdir(os.path.join(GROUP_DIR, 'results', 'run0', f'split{n_splits}')):
        n_splits += 1
    for r in range(3):
        for s in range(n_splits):
            ua = os.path.join(OUT_ROOT, f'run{r}', f'split{s}', 'summary.json')
            tr = os.path.join(GROUP_DIR, 'results', f'run{r}', f'split{s}')
            if not os.path.exists(ua):
                missing.append((r, s)); continue
            u = json.load(open(ua))
            rows.append({'variant': 'Untrained VQC', 'run_id': r, 'split_id': s,
                         'ood_family': u['ood_family'],
                         'ROC_raw': u['Threshold_Free']['roc_auc'], 'ROC_mad': u['roc_auc_mad_score'],
                         'PR_raw': u['Threshold_Free']['pr_auc'], 'TF_F1': u['Threshold_Free']['best_f1'],
                         'MAD_F1': u['MAD_Threshold']['best_f1']})
            try:
                t = json.load(open(os.path.join(tr, 'summary.json')))
                sc = list(csv.DictReader(open(os.path.join(tr, 'ood_scores.csv'))))
                mad_t = np.nan_to_num(np.array([float(x['mad_score']) for x in sc]), nan=0.0, posinf=1e12)
                _, y_tr, _, y_te, _ = load_split(data_dir, prefix, s)
                ood = (~np.isin(y_te, np.unique(y_tr))).astype(int)
                rows.append({'variant': 'Trained Group17', 'run_id': r, 'split_id': s,
                             'ood_family': u['ood_family'],
                             'ROC_raw': t['Threshold_Free']['roc_auc'],
                             'ROC_mad': float(roc_auc_score(ood, mad_t)),
                             'PR_raw': t['Threshold_Free']['pr_auc'], 'TF_F1': t['Threshold_Free']['best_f1'],
                             'MAD_F1': t['MAD_Threshold']['best_f1']})
            except Exception as e:
                log(f'  run{r}/split{s}: trained Group17 reference unavailable ({e})')
    log(f'Untrained tasks found: {len([x for x in rows if x["variant"] == "Untrained VQC"])}/'
        f'{3 * n_splits}   Missing: {missing if missing else "none"}')

    fams = []
    for x in rows:
        if x['ood_family'] not in fams:
            fams.append(x['ood_family'])
    mets = ['ROC_raw', 'ROC_mad', 'PR_raw', 'TF_F1', 'MAD_F1']
    log('Mean +- std over runs (population std). Flag-all F1 floor = 0.769.')
    for fam in fams + ['ALL']:
        log(f'\n  {fam}')
        log(f"    {'variant':<20}" + ''.join(f'{m:>15}' for m in mets))
        for v in ['Untrained VQC', 'Trained Group17']:
            sel = [x for x in rows if x['variant'] == v and (fam == 'ALL' or x['ood_family'] == fam)]
            if not sel:
                continue
            log(f'    {v:<20}' + ''.join(
                f'{np.mean([x[m] for x in sel]):>8.3f}+-{np.std([x[m] for x in sel]):<5.3f}' for m in mets))
        u = [x for x in rows if x['variant'] == 'Untrained VQC' and (fam == 'ALL' or x['ood_family'] == fam)]
        t = [x for x in rows if x['variant'] == 'Trained Group17' and (fam == 'ALL' or x['ood_family'] == fam)]
        if u and t and len(u) == len(t):
            log(f"    {'training gain':<20}" + ''.join(
                f'{np.mean([b[m] - a[m] for a, b in zip(u, t)]):>+15.3f}' for m in mets))

    keys = ['variant', 'run_id', 'split_id', 'ood_family'] + mets
    with open(os.path.join(HERE, 'aggregate.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    with open(os.path.join(HERE, 'aggregate_summary.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nSaved: {HERE}/aggregate.csv, aggregate_summary.txt')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('task_id', help="0-11, or 'aggregate'")
    ap.add_argument('--data-dir', default='~/projects/OOD/Data/cade_lamda_4class_vt')
    ap.add_argument('--prefix', default='drebin_new')
    args = ap.parse_args()
    data_dir = os.path.expanduser(args.data_dir)
    if args.task_id == 'aggregate':
        aggregate(data_dir, args.prefix)
    else:
        run_task(int(args.task_id), data_dir, args.prefix)


if __name__ == '__main__':
    main()
