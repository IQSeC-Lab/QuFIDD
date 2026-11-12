#!/usr/bin/env python3
"""seed_std_all16.py -- run from Experiments2 (via run_seed_std_all16.slurm).
Seed-wise aggregation for Table I, all 16 models x 3 datasets:
  1. per task (run, split): ROC-AUC, ranked F1, FPR and effort at the ranked-F1 stopping point (same code as before)
  2. per run (seed): mean over the 8 splits
  3. report mean and sample standard deviation (ddof=1) over the 3 seeds
The mean equals the old mean over 24 tasks; it is printed next to the old 24-task std for comparison.
Output: SEED_STD_ALL16.csv"""
import os, glob, csv
from collections import defaultdict
import numpy as np
from scipy.stats import rankdata

DATASETS = ['EMBER', 'Drebin', 'lamda']
MODELS = [
    ('Q_p', 'ablation_objective/quantum_hybrid_v7_p702_paironly'),
    ('Q_pp', 'quantum_pca8/pennylane/quantum_hybrid_v7_p702'),
    ('T_c-p', 'ablation_objective/classical_fidtwin_c_702_paironly'),
    ('T_pl-p', 'ablation_objective/classical_fidtwin_p_702_paironly'),
    ('T_r-p', 'ablation_objective/classical_fidtwin_r_702_paironly'),
    ('T_c-pp', 'twin_pca8_p702/classical_fidtwin_c_702'),
    ('T_pl-pp', 'twin_pca8_p702/classical_fidtwin_p_702'),
    ('T_r-pp', 'twin_pca8_p702/classical_fidtwin_r_702'),
    ('CADE_full-rp', 'cade_pca8_p702/cade_full_8_702'),
    ('CADE_enc-p', 'cade_pca8_p702/cade_enc_8_702'),
    ('CADE_tun-p', 'cade_pca8_p702/cade_tuned_8_702'),
    ('CADE_deep-p', 'cade_pca8_p702/cade_deep_8_702'),
    ('CADE_full-rpp', 'cade_pca8_p702_proto/cade_full_8_702_proto/tau_10'),
    ('CADE_enc-pp', 'cade_pca8_p702_proto/cade_enc_8_702_proto/tau_1'),
    ('CADE_tun-pp', 'cade_pca8_p702_proto/cade_tuned_8_702_proto/tau_10'),
    ('CADE_deep-pp', 'cade_pca8_p702_proto/cade_deep_8_702_proto/tau_1'),
]
METRICS = ['ROC', 'F1', 'FPR', 'Effort']


def task_metrics(f):
    rows = list(csv.DictReader(open(f)))
    s = np.array([float(r['raw_score']) for r in rows]); y = np.array([int(float(r['is_ood'])) for r in rows])
    P = int(y.sum()); N = len(y) - P
    r = rankdata(s)
    roc = (r[y == 1].sum() - P * (P + 1) / 2) / (P * N)
    ys = y[np.argsort(-s, kind='stable')]
    tp = np.cumsum(ys); k = np.arange(1, len(ys) + 1)
    f1 = 2 * tp / (k + P); b = int(np.argmax(f1))
    return {'ROC': roc * 100, 'F1': f1[b] * 100, 'FPR': (k[b] - tp[b]) / N * 100, 'Effort': k[b] / P}


rows = []
for ds in DATASETS:
    print(f'=== {ds}')
    print(f'  {"model":14s} {"ROC mean":>8s} {"seed sd":>7s} {"(24-task sd)":>12s}   {"F1 mean":>7s} {"seed sd":>7s} {"(24-task sd)":>12s}   '
          f'{"FPR":>5s} {"sd":>4s}  {"Effort":>6s} {"sd":>5s}  runs x splits')
    for name, fold in MODELS:
        files = sorted(glob.glob(os.path.join(ds, 'pca8', fold, 'results', 'run*', 'split*', 'ood_scores.csv')))
        if not files:
            print(f'  {name:14s} no ood_scores.csv in {fold}'); continue
        by_run = defaultdict(list)
        for f in files:
            by_run[os.path.basename(os.path.dirname(os.path.dirname(f)))].append(task_metrics(f))
        runs = sorted(by_run)
        per_run = {m: np.array([np.mean([t[m] for t in by_run[r]]) for r in runs]) for m in METRICS}
        alltasks = {m: np.array([t[m] for r in runs for t in by_run[r]]) for m in METRICS}
        res = {m: (per_run[m].mean(), per_run[m].std(ddof=1) if len(runs) > 1 else float('nan')) for m in METRICS}
        shape = 'x'.join(str(len(by_run[r])) for r in runs)
        print(f'  {name:14s} {res["ROC"][0]:8.1f} {res["ROC"][1]:7.1f} {alltasks["ROC"].std():12.1f}   '
              f'{res["F1"][0]:7.1f} {res["F1"][1]:7.1f} {alltasks["F1"].std():12.1f}   '
              f'{res["FPR"][0]:5.1f} {res["FPR"][1]:4.1f}  {res["Effort"][0]:6.2f} {res["Effort"][1]:5.2f}  {len(runs)} runs: {shape}')
        rows.append([ds, name, fold, len(runs), shape] + [f'{res[m][i]:.4f}' for m in METRICS for i in (0, 1)])

with open('SEED_STD_ALL16.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['dataset', 'model', 'folder', 'n_runs', 'splits_per_run'] + [f'{m}_{s}' for m in METRICS for s in ('mean', 'seed_sd')])
    w.writerows(rows)
print('\nmean = mean over 3 seeds of the 8-split average (equals the 24-task mean); seed sd = sample std (ddof=1) over seeds')
print('Saved SEED_STD_ALL16.csv')
