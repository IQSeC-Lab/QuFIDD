#!/usr/bin/env python3
"""quantum_vs_twin_pvalues.py -- run from Experiments2 (via run_quantum_vs_twin.slurm)
Impact of the circuit: quantum detector vs. each quantum-inspired twin under the SAME objective.
  CADE objective (pair loss only): {ds}/pca8/ablation_objective/<model>_paironly   (as in ablation_pvalues.py)
  full objective (pair + proto)  : the main folders in MAIN below
ROC-AUC per task (run, split), paired over the tasks both models share; two-sided paired Wilcoxon test.
'best twin' = the twin with the highest mean ROC-AUC on that dataset and objective."""
import os, glob, csv
import numpy as np
from scipy.stats import rankdata, wilcoxon

MAIN = {'Quantum': 'quantum_pca8/pennylane/quantum_hybrid_v7_p702', 'Twin, complex': 'twin_pca8_p702/classical_fidtwin_c_702',
        'Twin, plain': 'twin_pca8_p702/classical_fidtwin_p_702', 'Twin, real': 'twin_pca8_p702/classical_fidtwin_r_702'}
TWINS = [k for k in MAIN if k.startswith('Twin')]


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


def folders(ds, objective):
    """label -> folder for one objective; None when the folder is missing."""
    if objective == 'full':
        return {lab: f'{ds}/pca8/{m}' for lab, m in MAIN.items()}
    abl = sorted(glob.glob(f'{ds}/pca8/ablation_objective/*/'))
    out = {}
    for lab, m in MAIN.items():
        base = os.path.basename(m)
        hits = [a for a in abl if base in os.path.basename(a.rstrip('/')) and 'paironly' in os.path.basename(a.rstrip('/'))]
        out[lab] = hits[0] if hits else None
    return out


def compare(Q, T):
    keys = sorted(set(Q) & set(T))
    if len(keys) < 2:
        return None
    q = np.array([Q[k] for k in keys]); t = np.array([T[k] for k in keys])
    p = wilcoxon(q, t).pvalue if np.any(q != t) else 1.0
    return len(keys), q.mean(), t.mean(), q.mean() - t.mean(), int((q > t).sum()), p


rows_out = []
for ds in ['EMBER', 'Drebin', 'lamda']:
    for objective, name in [('cade', 'CADE objective (pair loss only)'), ('full', 'full objective (pair + prototype)')]:
        F = folders(ds, objective)
        print(f'=== {ds}  {name}')
        if F['Quantum'] is None:
            print('  quantum folder not found'); continue
        Q = task_aucs(F['Quantum'])
        res = {}
        for tw in TWINS:
            if F[tw] is None:
                print(f'  {tw:15s} folder not found'); continue
            r = compare(Q, task_aucs(F[tw]))
            if r is None:
                print(f'  {tw:15s} no paired tasks'); continue
            res[tw] = r
            n, qm, tm, d, wins, p = r
            print(f'  Quantum vs {tw:15s} n={n:2d}  quantum {qm:.3f}  twin {tm:.3f}  diff {d:+.3f}  '
                  f'quantum wins {wins}/{n}  p={p:.2g}')
            rows_out.append([ds, objective, tw, n, f'{qm:.4f}', f'{tm:.4f}', f'{d:+.4f}', wins, f'{p:.3g}'])
        if res:
            best = max(res, key=lambda k: res[k][2])
            n, qm, tm, d, wins, p = res[best]
            print(f'  -> best twin = {best}: diff {d:+.3f}, quantum wins {wins}/{n}, p={p:.2g}')

with open('QUANTUM_VS_TWIN_PVALUES.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['dataset', 'objective', 'twin', 'n_tasks', 'quantum_rocauc', 'twin_rocauc', 'diff', 'quantum_wins', 'p_wilcoxon'])
    w.writerows(rows_out)
print('\nSaved QUANTUM_VS_TWIN_PVALUES.csv')
