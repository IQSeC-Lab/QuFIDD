#!/usr/bin/env python3
"""collect_models.py -- one table per dataset of every trained model under <ds>/pca8; writes MODEL_RESULTS_ALL.txt/.csv"""
import os, glob, json, csv
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DATASETS = ['EMBER', 'Drebin', 'lamda']
METRICS = [('ROC-AUC', 'ROC_AUC'), ('PR-AUC', 'PR_AUC'), ('rankedF1', 'F1'), ('FPR', 'FPR'), ('Prec', 'Precision'), ('Recall', 'Recall')]


def group(rel):
    r = rel.lower()
    if 'ablation_objective' in r: return '6 ablation'
    if 'quantum' in r: return '1 quantum'
    if 'twin' in r: return '2 twin'
    if 'cade' in r: return '3 cade-mini'
    if 'pca' in r: return '4 pca-only'
    return '5 other'


rows, L = [], []
for ds in DATASETS:
    base = os.path.join(ROOT, ds, 'pca8')
    models = {}
    for f in glob.glob(os.path.join(base, '**', 'results', 'run*', 'split*', 'summary.json'), recursive=True):
        mdir = f.split(os.sep + 'results' + os.sep)[0]
        rel = os.path.relpath(mdir, base)
        if 'drift_analysis' in rel or 'explain' in rel:
            continue
        try:
            tf = json.load(open(f))['evaluation']['threshold_free']
        except Exception:
            continue
        models.setdefault(rel, []).append(tf)
    L.append('=' * 132); L.append(ds + '  (mean +- std over tasks; raw score, no filter)'); L.append('=' * 132)
    L.append('%-12s %-58s %3s ' % ('group', 'model', 'n') + ' '.join('%13s' % m for m, _ in METRICS))
    for rel in sorted(models, key=lambda r: (group(r), r)):
        T = models[rel]; vals = {}
        for m, k in METRICS:
            v = [t[k] for t in T if k in t and t[k] is not None]
            vals[m] = (np.mean(v), np.std(v)) if v else (np.nan, np.nan)
        L.append('%-12s %-58s %3d ' % (group(rel)[2:], rel[-58:], len(T)) +
                 ' '.join('%6.3f+-%.3f' % vals[m] for m, _ in METRICS))
        rows.append(dict([('dataset', ds), ('group', group(rel)[2:]), ('model', rel), ('n_tasks', len(T))] +
                         [(m + '_mean', round(vals[m][0], 4)) for m, _ in METRICS] + [(m + '_std', round(vals[m][1], 4)) for m, _ in METRICS]))
    L.append('')
L.append('n = finished tasks (24 expected). rankedF1/FPR/Prec/Recall at the best ranked-F1 stopping point (CADE protocol, T = 0).')
open(os.path.join(ROOT, 'MODEL_RESULTS_ALL.txt'), 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
if rows:
    with open(os.path.join(ROOT, 'MODEL_RESULTS_ALL.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
