#!/usr/bin/env python3
"""perfamily_groups.py -- run from Experiments2 (via run_perfamily_groups.slurm).
Per-family ROC-AUC (one withheld family per split, mean over the three runs) for the six models of the new heatmap:
both quantum versions and the best twin and CADE-mini of each loss setting per dataset (same models as Figs. 3 and 4).
Prints the split -> family mapping source (PERFAMILY_AUC.csv header) so the families can be matched to the splits.
Output: PERFAMILY_GROUPS.csv"""
import os, glob, csv
import numpy as np
from scipy.stats import rankdata

Q_PAIR = 'ablation_objective/quantum_hybrid_v7_p702_paironly'
Q_FULL = 'quantum_pca8/pennylane/quantum_hybrid_v7_p702'
TW = {c: f'twin_pca8_p702/classical_fidtwin_{c}_702' for c in 'cpr'}
TWP = {c: f'ablation_objective/classical_fidtwin_{c}_702_paironly' for c in 'cpr'}
CP = {'enc': 'cade_pca8_p702_proto/cade_enc_8_702_proto/tau_1', 'tuned': 'cade_pca8_p702_proto/cade_tuned_8_702_proto/tau_10',
      'deep': 'cade_pca8_p702_proto/cade_deep_8_702_proto/tau_1'}
C = {v: f'cade_pca8_p702/cade_{v}_8_702' for v in ('full', 'enc')}
ROWS = [
    ('Quantum (pair)', {'EMBER': Q_PAIR, 'Drebin': Q_PAIR, 'lamda': Q_PAIR}),
    ('Quantum (pair + prototype)', {'EMBER': Q_FULL, 'Drebin': Q_FULL, 'lamda': Q_FULL}),
    ('Twin (pair)', {'EMBER': TWP['r'], 'Drebin': TWP['c'], 'lamda': TWP['c']}),
    ('Twin (pair + prototype)', {'EMBER': TW['c'], 'Drebin': TW['c'], 'lamda': TW['p']}),
    ('CADE-mini (pair)', {'EMBER': C['enc'], 'Drebin': C['enc'], 'lamda': C['full']}),
    ('CADE-mini (pair + prototype)', {'EMBER': CP['enc'], 'Drebin': CP['tuned'], 'lamda': CP['deep']}),
]


def auc(f):
    rows = list(csv.DictReader(open(f)))
    s = np.array([float(r['raw_score']) for r in rows]); y = np.array([int(float(r['is_ood'])) for r in rows])
    P = y.sum(); N = len(y) - P; r = rankdata(s)
    return (r[y == 1].sum() - P * (P + 1) / 2) / (P * N)


out = []
for ds in ['EMBER', 'Drebin', 'lamda']:
    print(f'=== {ds}')
    for lab, folders in ROWS:
        base = os.path.join(ds, 'pca8', folders[ds], 'results')
        splits = sorted({os.path.basename(os.path.dirname(f)) for f in glob.glob(os.path.join(base, 'run*', 'split*', 'ood_scores.csv'))},
                        key=lambda s: int(''.join(ch for ch in s if ch.isdigit()) or 0))
        vals = []
        for sp in splits:
            a = [auc(f) for f in sorted(glob.glob(os.path.join(base, 'run*', sp, 'ood_scores.csv')))]
            vals.append(np.mean(a)); out.append([ds, lab, folders[ds], sp, len(a), f'{np.mean(a):.4f}'])
        print(f'  {lab:30s} ' + ' '.join(f'{sp}:{v:.2f}' for sp, v in zip(splits, vals)))

with open('PERFAMILY_GROUPS.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['dataset', 'model', 'folder', 'split', 'n_runs', 'roc_auc']); w.writerows(out)
print('Saved PERFAMILY_GROUPS.csv')

for ref in ['PERFAMILY_AUC.csv']:
    if os.path.exists(ref):
        print(f'\n--- {ref} (first 12 lines, for the split -> family mapping)')
        for i, l in enumerate(open(ref)):
            if i >= 12: break
            print('  ' + l.rstrip())
# family names per split, if the split folders record them
for ds in ['EMBER', 'Drebin', 'lamda']:
    d = glob.glob(os.path.join(ds, 'pca8', Q_FULL, 'results', 'run0', 'split*'))
    if d:
        sp = sorted(d)[0]
        print(f'\n--- files in {sp}: ' + ' '.join(sorted(os.listdir(sp))))
