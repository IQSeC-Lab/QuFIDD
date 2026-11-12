#!/usr/bin/env python3
"""ablation_pvalues.py -- run from Experiments2:  python3 ablation_pvalues.py
For each model: ROC-AUC per task (run, split) under the CADE objective and the full objective, paired Wilcoxon test.
The main folders hold one objective; ablation_objective/ holds the other (matched by folder base name)."""
import os, glob, csv
import numpy as np
from scipy.stats import rankdata, wilcoxon

MAIN = {'Quantum': 'quantum_pca8/pennylane/quantum_hybrid_v7_p702', 'Twin, complex': 'twin_pca8_p702/classical_fidtwin_c_702',
        'Twin, plain': 'twin_pca8_p702/classical_fidtwin_p_702', 'Twin, real': 'twin_pca8_p702/classical_fidtwin_r_702',
        'CADE-mini, encoder': 'cade_pca8_p702/cade_enc_8_702'}
MAIN_OBJ = {'Quantum': 'full', 'Twin, complex': 'full', 'Twin, plain': 'full', 'Twin, real': 'full', 'CADE-mini, encoder': 'cade'}


def auc(s, y):
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def task_aucs(folder):
    out = {}
    for f in glob.glob(os.path.join(folder, 'results', 'run*', 'split*', 'ood_scores.csv')):
        rows = list(csv.DictReader(open(f)))
        s = np.array([float(r['raw_score']) for r in rows]); y = np.array([int(float(r['is_ood'])) for r in rows])
        parts = f.split(os.sep); out[(parts[-3], parts[-2])] = auc(s, y)
    return out


for ds in ['EMBER', 'Drebin', 'lamda']:
    abl = sorted(glob.glob(f'{ds}/pca8/ablation_objective/*/'))
    print(f'=== {ds}   (ablation folders: {[os.path.basename(a.rstrip("/")) for a in abl]})')
    for lab, m in MAIN.items():
        base = os.path.basename(m)
        hits = [a for a in abl if base in os.path.basename(a.rstrip('/'))]
        if not hits:
            print(f'  {lab:20s} no ablation folder containing "{base}"'); continue
        A, B = task_aucs(f'{ds}/pca8/{m}'), task_aucs(hits[0])
        keys = sorted(set(A) & set(B))
        if not keys:
            print(f'  {lab:20s} no paired tasks ({len(A)} main, {len(B)} ablation)'); continue
        a = np.array([A[k] for k in keys]); b = np.array([B[k] for k in keys])
        cade, full = (b, a) if MAIN_OBJ[lab] == 'full' else (a, b)
        p = wilcoxon(full, cade).pvalue
        print(f'  {lab:20s} n={len(keys):2d}  CADE obj {cade.mean():.3f}  full obj {full.mean():.3f}  change {full.mean() - cade.mean():+.3f}  '
              f'p={p:.2g}  [{os.path.basename(hits[0].rstrip("/"))}]')
