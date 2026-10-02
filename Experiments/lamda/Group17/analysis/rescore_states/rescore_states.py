#!/usr/bin/env python3
"""
rescore_states.py  --  Group17: score with the information the current score ignores
====================================================================================
No retraining. Group17's current OOD score uses only 3 of the 32 measurement
probabilities p = |psi|^2 (those on the known classes' target basis states):
    current   = min_c (1 - p_c) = 1 - max_c p_c
Training pushes KNOWN families into their targets; a NEW family has no target,
so its probability should leak into the other basis states. This script adds
scores that use the whole distribution:
    leakage   = 1 - sum_{c known} p_c          (mass outside all known targets)
    prob_eucl = min_c || p - mu_c ||_2          (mu_c = mean p of class c, training)
    hellinger = min_c || sqrt(p) - sqrt(mu_c) ||_2 / sqrt(2)
All are functions of measurement probabilities in a FIXED basis, so they are
not cancelled by unitary invariance (unlike fidelity to class density
matrices, which is theta-independent and deliberately NOT used here).
Each score also gets a MAD version (per-class median/MAD of training
distances, b = 1.4826, floor 1e-6; leakage uses the pooled training
distribution) and a fixed-threshold (3.5) MAD F1.

Per task it:
  1. Rebuilds Group17's model (VQCEncoder, same seed) and loads the TRAINED
     weights (../../results/run{r}/split{s}/model_weights.pt), or keeps the
     initial weights with --untrained; loads the saved PCA pickle.
  2. Simulates states for X_train / X_test and SAVES them
     (states_trained.npz / states_untrained.npz) so later re-scoring needs no
     simulation. If the states file already exists it is reused.
  3. Sanity check: the 'current' score must reproduce the saved
     ood_scores.csv raw_score of trained Group17.
  4. Evaluates every score with ROC-AUC, PR-AUC, threshold-free best F1 and
     MAD F1, and compares against PCA-only when its results exist
     (../../ablation/pca_only/results/).

Layout:
  Group17/
    Group17_vqc_fixed_target_fidelity_4class.py
    results/run*/split*/{model_weights.pt, pca_components.pkl, ood_scores.csv}
    ablation/pca_only/results/...                 (optional, for comparison)
    analysis/rescore_states/
      rescore_states.py                           <- this script
      run_rescore_states.slurm
      trained/run*/split*/{states_trained.npz, rescore.json}
      untrained/run*/split*/{states_untrained.npz, rescore.json}
      aggregate_trained.csv / aggregate_summary_trained.txt (and _untrained)

Usage (from Group17/analysis/rescore_states/, OOD_env active):
  python3 rescore_states.py <task_id 0-11>                 # trained weights
  python3 rescore_states.py <task_id 0-11> --untrained     # initial weights
  python3 rescore_states.py aggregate [--untrained]
"""

import os, sys, csv, json, time, pickle, argparse, importlib.util, warnings
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

HERE      = os.path.dirname(os.path.abspath(__file__))
GROUP_DIR = os.path.dirname(os.path.dirname(HERE))
G17_FILE  = os.path.join(GROUP_DIR, 'Group17_vqc_fixed_target_fidelity_4class.py')
PCA_ONLY  = os.path.join(GROUP_DIR, 'ablation', 'pca_only', 'results')
SCORES    = ['current', 'leakage', 'prob_eucl', 'hellinger']
MAD_T     = 3.5
N_SPLITS  = 4
SEEDS     = {0: 42, 1: 123, 2: 456}


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
    return dense(d['X_train']), d['y_train'].astype(np.int64), dense(d['X_test']), d['y_test'].astype(np.int64), fam


# ---------------------------------------------------------------- scoring
def distances(p_tr, y_tr, p_te):
    """Returns {score: (train_dist_to_own_class_or_pooled, test_dist_matrix_or_vector, per_class)}."""
    classes = sorted(int(c) for c in np.unique(y_tr))
    mu = {c: p_tr[y_tr == c].mean(axis=0) for c in classes}
    out = {}
    # current: 1 - p_c
    out['current'] = ({c: 1.0 - p_tr[y_tr == c, c] for c in classes},
                      np.stack([1.0 - p_te[:, c] for c in classes], axis=1), classes)
    # leakage: pooled (one 'class')
    known = np.array(classes)
    out['leakage'] = ({'all': 1.0 - p_tr[:, known].sum(axis=1)},
                      (1.0 - p_te[:, known].sum(axis=1))[:, None], ['all'])
    # prob_eucl
    out['prob_eucl'] = ({c: np.linalg.norm(p_tr[y_tr == c] - mu[c], axis=1) for c in classes},
                        np.stack([np.linalg.norm(p_te - mu[c], axis=1) for c in classes], axis=1), classes)
    # hellinger
    sq_tr, sq_te = np.sqrt(p_tr), np.sqrt(p_te)
    out['hellinger'] = ({c: np.linalg.norm(sq_tr[y_tr == c] - np.sqrt(mu[c]), axis=1) / np.sqrt(2) for c in classes},
                        np.stack([np.linalg.norm(sq_te - np.sqrt(mu[c]), axis=1) / np.sqrt(2) for c in classes], axis=1),
                        classes)
    return out


def raw_and_mad(train_d, test_D, keys):
    raw = test_D.min(axis=1)
    A = []
    for j, k in enumerate(keys):
        med = np.median(train_d[k])
        mad = max(1.4826 * np.median(np.abs(train_d[k] - med)), 1e-6)
        A.append(np.abs(test_D[:, j] - med) / mad)
    return raw, np.stack(A, axis=1).min(axis=1)


def best_f1(scores, is_ood):
    order = np.argsort(-scores, kind='stable')
    tp = np.cumsum(is_ood[order]); k = np.arange(1, len(order) + 1)
    p, r = tp / k, tp / is_ood.sum()
    return float(np.max(np.where(p + r > 0, 2 * p * r / (p + r + 1e-12), 0.0)))


def mad_f1(mad, is_ood, T=MAD_T):
    pred = mad > T
    tp = int((pred & (is_ood == 1)).sum()); fp = int((pred & (is_ood == 0)).sum())
    fn = int(((~pred) & (is_ood == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
    return (2 * p * r / (p + r) if p + r else 0.0), float(pred.mean())


def evaluate(raw, mad, is_ood):
    f1m, frac = mad_f1(mad, is_ood)
    return {'ROC_raw': float(roc_auc_score(is_ood, raw)), 'PR_raw': float(average_precision_score(is_ood, raw)),
            'TF_F1': best_f1(raw, is_ood), 'ROC_mad': float(roc_auc_score(is_ood, mad)),
            'MAD_F1': float(f1m), 'MAD_frac_flagged': frac}


# ---------------------------------------------------------------- per task
def run_task(task_id, data_dir, prefix, untrained):
    r, s = task_id // N_SPLITS, task_id % N_SPLITS
    seed = SEEDS[r]
    tag = 'untrained' if untrained else 'trained'
    X_tr, y_tr, X_te, y_te, fam = load_split(data_dir, prefix, s)
    is_ood = (~np.isin(y_te, np.unique(y_tr))).astype(int)
    src = os.path.join(GROUP_DIR, 'results', f'run{r}', f'split{s}')
    out = os.path.join(HERE, tag, f'run{r}', f'split{s}')
    os.makedirs(out, exist_ok=True)
    st_path = os.path.join(out, f'states_{tag}.npz')
    print('=' * 70, flush=True)
    print(f'Group17 RESCORE ({tag}) | task={task_id} run={r} split={s} seed={seed} OOD={fam}', flush=True)
    print('=' * 70, flush=True)

    if os.path.exists(st_path):
        z = np.load(st_path); st_tr, st_te = z['states_train'], z['states_test']
        print(f'  Reusing saved states: {st_path}', flush=True); sim_time = 0.0
    else:
        import torch
        g17 = load_g17()
        g17.set_seed(seed)
        model = g17.VQCEncoder(X_tr.shape[1], seed=seed)
        if not untrained:
            sd = torch.load(os.path.join(src, 'model_weights.pt'), map_location='cpu')
            model.load_state_dict(sd)
        pkl = os.path.join(src, 'pca_components.pkl')
        if os.path.exists(pkl):
            model.pca_comp.pca = pickle.load(open(pkl, 'rb')); model.pca_comp.fitted = True
        else:
            model.pca_comp.fit(X_tr); print('  PCA pickle missing -> refit', flush=True)
        print(f'  q_weights mean|w| = {model.q_weights.detach().abs().mean().item():.5f}', flush=True)
        t0 = time.time()
        st_tr = model.get_states_numpy(X_tr); st_te = model.get_states_numpy(X_te)
        sim_time = time.time() - t0
        np.savez_compressed(st_path, states_train=st_tr, states_test=st_te,
                            y_train=y_tr, y_test=y_te)
        print(f'  Simulated + saved states ({sim_time:.1f}s): {st_path}', flush=True)

    p_tr = np.abs(st_tr) ** 2; p_te = np.abs(st_te) ** 2
    D = distances(p_tr, y_tr, p_te)
    res = {}
    for name in SCORES:
        raw, mad = raw_and_mad(*D[name])
        res[name] = evaluate(raw, mad, is_ood)
        print(f'  {name:<10} ROC_raw={res[name]["ROC_raw"]:.3f}  PR={res[name]["PR_raw"]:.3f}  '
              f'TF_F1={res[name]["TF_F1"]:.3f}  ROC_mad={res[name]["ROC_mad"]:.3f}  '
              f'MAD_F1={res[name]["MAD_F1"]:.3f} (flagged {res[name]["MAD_frac_flagged"]:.2f})', flush=True)
        if name == 'current' and not untrained:
            sc = os.path.join(src, 'ood_scores.csv')
            if os.path.exists(sc):
                saved = np.array([float(x['raw_score']) for x in csv.DictReader(open(sc))])
                diff = float(np.max(np.abs(saved - raw)))
                res['sanity_max_abs_diff_vs_saved_raw'] = diff
                print(f'  Sanity: current vs saved raw_score max|diff| = {diff:.2e} '
                      + ('OK' if diff < 1e-4 else '<-- CHECK'), flush=True)
    summary = {'group': 'Group17', 'weights': tag, 'run_id': r, 'split_id': s, 'seed': seed,
               'ood_family': fam, 'sim_time_sec': round(sim_time, 2), 'scores': res}
    with open(os.path.join(out, 'rescore.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'  Saved: {out}/rescore.json', flush=True)


# ---------------------------------------------------------------- aggregate
def aggregate(untrained):
    warnings.filterwarnings('ignore', category=RuntimeWarning)   # nan columns for the PCA-only reference row
    tag = 'untrained' if untrained else 'trained'
    lines = []
    def log(s=''):
        print(s, flush=True); lines.append(s)
    log('=' * 92)
    log(f'Group17 RESCORE ({tag} weights) -- scores using the full measurement distribution')
    log('=' * 92)
    rows, missing = [], []
    for r in range(3):
        for s in range(N_SPLITS):
            p = os.path.join(HERE, tag, f'run{r}', f'split{s}', 'rescore.json')
            if not os.path.exists(p):
                missing.append((r, s)); continue
            d = json.load(open(p))
            for name in SCORES:
                rows.append({'run_id': r, 'split_id': s, 'ood_family': d['ood_family'],
                             'score': name, **d['scores'][name]})
            po = os.path.join(PCA_ONLY, f'run{r}', f'split{s}', 'summary.json')
            if os.path.exists(po):
                q = json.load(open(po))
                rows.append({'run_id': r, 'split_id': s, 'ood_family': d['ood_family'],
                             'score': 'PCA-only (ref)', 'ROC_raw': q['Threshold_Free']['roc_auc'],
                             'PR_raw': q['Threshold_Free']['pr_auc'], 'TF_F1': q['Threshold_Free']['best_f1'],
                             'ROC_mad': q.get('roc_auc_mad_score', np.nan),
                             'MAD_F1': q['MAD_Threshold']['best_f1'], 'MAD_frac_flagged': np.nan})
            if 'sanity_max_abs_diff_vs_saved_raw' in d['scores']:
                rows[-5 if os.path.exists(po) else -4]['sanity'] = d['scores']['sanity_max_abs_diff_vs_saved_raw']
    n_done = len({(x['run_id'], x['split_id']) for x in rows})
    log(f'Tasks: {n_done}/{3 * N_SPLITS}   Missing: {missing if missing else "none"}')
    san = [x['sanity'] for x in rows if 'sanity' in x]
    if san:
        log(f'Sanity (current score reproduces saved Group17 raw_score): max|diff| = {max(san):.2e} '
            + ('OK' if max(san) < 1e-4 else '<-- CHECK'))
    fams = []
    for x in rows:
        if x['ood_family'] not in fams:
            fams.append(x['ood_family'])
    mets = ['ROC_raw', 'PR_raw', 'TF_F1', 'ROC_mad', 'MAD_F1', 'MAD_frac_flagged']
    names = SCORES + (['PCA-only (ref)'] if any(x['score'] == 'PCA-only (ref)' for x in rows) else [])
    agg = []
    log('Mean +- std over runs (population std). Flag-all F1 floor = 0.769.')
    for fam in fams + ['ALL']:
        log(f'\n  {fam}')
        log(f"    {'score':<16}" + ''.join(f'{m:>17}' for m in mets))
        for n in names:
            sel = [x for x in rows if x['score'] == n and (fam == 'ALL' or x['ood_family'] == fam)]
            if not sel:
                continue
            line, rec = f'    {n:<16}', {'score': n, 'ood_family': fam, 'n_tasks': len(sel)}
            for m in mets:
                a = np.array([x[m] for x in sel], float)
                mu, sd = np.nanmean(a), np.nanstd(a)
                line += f'{mu:>10.3f}+-{sd:<5.3f}'
                rec[f'{m}_mean'], rec[f'{m}_std'] = round(float(mu), 6), round(float(sd), 6)
            log(line); agg.append(rec)
    with open(os.path.join(HERE, f'aggregate_{tag}.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(agg[0].keys())); w.writeheader(); w.writerows(agg)
    with open(os.path.join(HERE, f'aggregate_summary_{tag}.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nSaved: {HERE}/aggregate_{tag}.csv, aggregate_summary_{tag}.txt')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('task_id', help="0-11, or 'aggregate'")
    ap.add_argument('--untrained', action='store_true', help='use initial (untrained) weights')
    ap.add_argument('--data-dir', default='~/projects/OOD/Data/cade_lamda_4class_vt')
    ap.add_argument('--prefix', default='drebin_new')
    args = ap.parse_args()
    if args.task_id == 'aggregate':
        aggregate(args.untrained)
    else:
        run_task(int(args.task_id), os.path.expanduser(args.data_dir), args.prefix, args.untrained)


if __name__ == '__main__':
    main()
