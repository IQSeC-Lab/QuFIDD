#!/usr/bin/env python3
"""compare_ablation.py -- 2x2 objective ablation (geometry x objective) at the p702 budget, mean +- std over the 24 tasks,
plus paired Wilcoxon tests on matched (run, split) tasks. Run from this ablation_objective folder after aggregate_all.slurm.
Writes compare_ablation.txt and compare_ablation.csv."""
import os, json, glob, csv
import numpy as np
from scipy.stats import wilcoxon
here = os.path.dirname(os.path.abspath(__file__))
P = lambda *a: os.path.join(here, *a)
# (label, geometry, objective, results dir)
MODELS = [
    ('Quantum hybrid', 'fidelity (quantum)', 'pair + proto', P('..', 'quantum_pca8', 'pennylane', 'quantum_hybrid_v7_p702', 'results')),
    ('Quantum hybrid', 'fidelity (quantum)', 'pair only', P('quantum_hybrid_v7_p702_paironly', 'results')),
    ('Twin, complex', 'fidelity (classical)', 'pair + proto', P('..', 'twin_pca8_p702', 'classical_fidtwin_c_702', 'results')),
    ('Twin, complex', 'fidelity (classical)', 'pair only', P('classical_fidtwin_c_702_paironly', 'results')),
    ('Twin, plain', 'fidelity (classical)', 'pair + proto', P('..', 'twin_pca8_p702', 'classical_fidtwin_p_702', 'results')),
    ('Twin, plain', 'fidelity (classical)', 'pair only', P('classical_fidtwin_p_702_paironly', 'results')),
    ('Twin, real', 'fidelity (classical)', 'pair + proto', P('..', 'twin_pca8_p702', 'classical_fidtwin_r_702', 'results')),
    ('Twin, real', 'fidelity (classical)', 'pair only', P('classical_fidtwin_r_702_paironly', 'results')),
    ('CADE-mini, encoder', 'Euclidean', 'pair only', P('..', 'cade_pca8_p702', 'cade_enc_8_702', 'results')),
    ('CADE-mini, encoder', 'Euclidean', 'pair + proto', P('cade_enc_8_702_proto', 'results')),
]
METRICS = [('ROC-AUC', lambda e: e['threshold_free']['ROC_AUC']), ('PR-AUC', lambda e: e['threshold_free']['PR_AUC']),
           ('ranked F1', lambda e: e['threshold_free']['F1'])]


def load(rd):
    out = {}
    for f in glob.glob(os.path.join(rd, 'run*', 'split*', 'summary.json')):
        r, s = f.split(os.sep)[-3:-1]; out[(r, s)] = json.load(open(f))['evaluation']
    return out


R = [(m, load(m[3])) for m in MODELS]
L = ['Objective ablation at the p702 budget (PCA-8 input; same splits, seeds, early stopping and scoring per model)',
     f"{'model':20s} {'geometry':22s} {'objective':13s} {'n':>3s} " + ' '.join(f'{k:>15s}' for k, _ in METRICS)]
rows = []
for (lab, geo, obj, _), E in R:
    vals = {k: [f(e) for e in E.values()] for k, f in METRICS}
    L.append(f'{lab:20s} {geo:22s} {obj:13s} {len(E):3d} ' + ' '.join(
        (f'{np.mean(v):.3f} +- {np.std(v):.3f}' if v else f'{"-":>15s}') for v in vals.values()))
    rows.append({'model': lab, 'geometry': geo, 'objective': obj, 'n': len(E),
                 **{f'{k}_mean': (np.mean(v) if v else None) for k, v in vals.items()},
                 **{f'{k}_std': (np.std(v) if v else None) for k, v in vals.items()}})


def paired(a, b, title):
    A, B = R[a][1], R[b][1]; keys = sorted(set(A) & set(B))
    if len(keys) < 6:
        return L.append(f'  {title}: too few matched tasks ({len(keys)})')
    for k, f in METRICS:
        x = np.array([f(A[t]) for t in keys]); y = np.array([f(B[t]) for t in keys])
        p = wilcoxon(x, y).pvalue if np.any(x != y) else 1.0
        L.append(f'  {title:62s} {k:9s} delta {np.mean(x - y):+.3f}  wins {int((x > y).sum())}/{len(keys)}  Wilcoxon p {p:.3g}')


L.append('\nPaired tests (first minus second, matched run/split):')
paired(0, 1, 'objective, quantum: pair+proto vs pair only')
paired(2, 3, 'objective, twin c: pair+proto vs pair only')
paired(9, 8, 'objective, Euclidean: pair+proto vs pair only')
paired(0, 9, 'geometry, our objective: quantum vs CADE-mini+proto')
paired(1, 8, 'geometry, CADE objective: quantum pair-only vs CADE-mini')
paired(3, 8, 'geometry, CADE objective: twin c pair-only vs CADE-mini')
paired(0, 2, 'quantum vs classical fidelity (both pair+proto)')
open(P('compare_ablation.txt'), 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
with open(P('compare_ablation.csv'), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
