#!/usr/bin/env python3
"""collect_cade_proto_tau.py -- run from Experiments2 (via run_collect_cade_proto_tau.slurm) after all arrays finish.

For every CADE-mini variant (full, encoder, tuned, deep), every dataset, and every temperature tau, computes from
the per-task ood_scores.csv files (raw score, CADE protocol, T = 0) the same metrics as Table I:
  ROC-AUC, PR-AUC, ranked F1, and at the ranked-F1 stopping point: precision, recall, FPR, inspection effort
  (effort = inspected samples / drifting samples), each as mean +- std over the tasks.
The same variant WITHOUT the prototype loss (cade_pca8_p702) is computed with the same code and compared with
MODEL_RESULTS_ALL.csv and the effort in Table I, so the computation can be checked before the new numbers are used.
The best tau per variant is the one with the highest ROC-AUC averaged over the datasets (one tau per model).

Outputs: CADE_PROTO_TAU_ALL.csv (every metric, every run), CADE_PROTO_BEST.csv (best tau per variant),
         printed summary in the log."""
import os, glob, csv
import numpy as np
from scipy.stats import rankdata

DATASETS = ['EMBER', 'Drebin', 'lamda']
VARIANTS = ['cade_full_8_702', 'cade_enc_8_702', 'cade_tuned_8_702', 'cade_deep_8_702']
METRICS = ['ROC-AUC', 'PR-AUC', 'rankedF1', 'Prec', 'Recall', 'FPR', 'Effort']
CSV_COL = {'ROC-AUC': 'ROC-AUC_mean', 'PR-AUC': 'PR-AUC_mean', 'rankedF1': 'rankedF1_mean',
           'Prec': 'Prec_mean', 'Recall': 'Recall_mean', 'FPR': 'FPR_mean'}
TABLE1_EFFORT = {  # effort of the variants without prototype loss in Table I (EMBER, Drebin, LAMDA)
    'cade_full_8_702': (1.82, 1.39, 1.71), 'cade_enc_8_702': (1.71, 1.40, 1.78),
    'cade_tuned_8_702': (1.69, 1.25, 1.73), 'cade_deep_8_702': (1.79, 1.40, 2.06)}


def task_metrics(f):
    rows = list(csv.DictReader(open(f)))
    s = np.array([float(r['raw_score']) for r in rows]); y = np.array([int(float(r['is_ood'])) for r in rows])
    P = int(y.sum()); N = len(y) - P
    r = rankdata(s)
    roc = (r[y == 1].sum() - P * (P + 1) / 2) / (P * N)
    order = np.argsort(-s, kind='stable'); ys = y[order]
    tp = np.cumsum(ys); k = np.arange(1, len(ys) + 1)
    prec_k = tp / k
    pr = float((prec_k * ys).sum() / P)            # average precision (area under the precision-recall curve)
    f1 = 2 * tp / (k + P)
    b = int(np.argmax(f1))                          # stopping point of the ranked F1
    return {'ROC-AUC': roc, 'PR-AUC': pr, 'rankedF1': f1[b], 'Prec': prec_k[b], 'Recall': tp[b] / P,
            'FPR': (k[b] - tp[b]) / N, 'Effort': k[b] / P}


def model_metrics(folder):
    files = sorted(glob.glob(os.path.join(folder, 'results', 'run*', 'split*', 'ood_scores.csv')))
    if not files:
        return 0, None
    t = [task_metrics(f) for f in files]
    return len(files), {m: (float(np.mean([x[m] for x in t])), float(np.std([x[m] for x in t]))) for m in METRICS}


def load_reference():
    ref = {}
    if os.path.exists('MODEL_RESULTS_ALL.csv'):
        for r in csv.DictReader(open('MODEL_RESULTS_ALL.csv')):
            ref[(r['dataset'], r['model'])] = r
    return ref


def fmt(res, m):
    return '      --     ' if res is None else f'{res[m][0]:.3f}+-{res[m][1]:.3f}'


ref = load_reference()
taus = sorted({os.path.basename(p)[4:] for p in glob.glob('*/pca8/cade_pca8_p702_proto/*/tau_*')}, key=float)
all_rows, best_rows = [], []

# ---------------------------------------------------------------- 1. check against existing results
print('=== CHECK: variants WITHOUT prototype loss, recomputed here vs MODEL_RESULTS_ALL.csv / Table I effort')
for di, ds in enumerate(DATASETS):
    for v in VARIANTS:
        n, res = model_metrics(f'{ds}/pca8/cade_pca8_p702/{v}')
        if res is None:
            print(f'  {ds:7s} {v:18s} no ood_scores.csv'); continue
        r = ref.get((ds, f'cade_pca8_p702/{v}'))
        parts = []
        for m in METRICS:
            if m == 'Effort':
                parts.append(f'Effort {res[m][0]:.2f}/{TABLE1_EFFORT[v][di]:.2f}')
            elif r is not None:
                parts.append(f'{m} {res[m][0]:.3f}/{float(r[CSV_COL[m]]):.3f}')
        print(f'  {ds:7s} {v:18s} n={n:2d}  ' + '  '.join(parts) + ('' if r else '  (not in MODEL_RESULTS_ALL.csv)'))
print('  (each pair = recomputed here / existing value; they should agree to about 0.001, effort to about 0.01)\n')

# ---------------------------------------------------------------- 2. all runs
for v in VARIANTS:
    print(f'=== {v}')
    base = {ds: model_metrics(f'{ds}/pca8/cade_pca8_p702/{v}') for ds in DATASETS}
    runs = {t: {ds: model_metrics(f'{ds}/pca8/cade_pca8_p702_proto/{v}_proto/tau_{t}') for ds in DATASETS} for t in taus}
    for ds in DATASETS:
        print(f'  -- {ds}')
        print(f'    {"":10s}' + ''.join(f'{m:>15s}' for m in METRICS) + '   n')
        for label, (n, res) in [('no proto', base[ds])] + [(f'tau={t}', runs[t][ds]) for t in taus]:
            print(f'    {label:10s}' + ''.join(f'{fmt(res, m):>15s}' for m in METRICS) + f'  {n:2d}')
            if res is not None:
                all_rows.append([v, label.replace('tau=', ''), ds, n] +
                                [f'{res[m][i]:.4f}' for m in METRICS for i in (0, 1)])
    complete = {t: np.mean([runs[t][ds][1]['ROC-AUC'][0] for ds in DATASETS])
                for t in taus if all(runs[t][ds][0] == 24 for ds in DATASETS)}
    if complete:
        bt = max(complete, key=complete.get)
        print(f'  -> best tau = {bt} (ROC-AUC averaged over datasets {complete[bt]:.3f})')
        for ds in DATASETS:
            res = runs[bt][ds][1]
            best_rows.append([v, bt, ds] + [f'{res[m][i]:.4f}' for m in METRICS for i in (0, 1)])
    else:
        print('  -> no tau finished on all three datasets yet (24 tasks each)')
    print()

hdr = [f'{m}_{s}' for m in METRICS for s in ('mean', 'std')]
with open('CADE_PROTO_TAU_ALL.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['variant', 'tau', 'dataset', 'n_tasks'] + hdr); w.writerows(all_rows)
with open('CADE_PROTO_BEST.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['variant', 'best_tau', 'dataset'] + hdr); w.writerows(best_rows)
print('Saved CADE_PROTO_TAU_ALL.csv (every run) and CADE_PROTO_BEST.csv (best tau per variant)')
