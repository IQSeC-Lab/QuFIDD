#!/usr/bin/env python3
"""perfamily_auc.py -- run from Experiments2 via SLURM. Per-family ROC-AUC of all nine models on all three datasets,
computed from ood_scores.csv (raw_score, is_ood) and averaged over the three seeds of each setting.
Writes PERFAMILY_AUC.csv (dataset, split, family, model, auc_mean, auc_std, n_seeds) and prints one table per dataset."""
import os, glob, csv, json
import numpy as np
from scipy.stats import rankdata

MODELS = [('Quantum (pair loss only)', 'ablation_objective/quantum_hybrid_v7_p702_paironly'),
          ('Quantum (pair + prototype)', 'quantum_pca8/pennylane/quantum_hybrid_v7_p702'),
          ('Twin, complex', 'twin_pca8_p702/classical_fidtwin_c_702'), ('Twin, plain', 'twin_pca8_p702/classical_fidtwin_p_702'),
          ('Twin, real', 'twin_pca8_p702/classical_fidtwin_r_702'), ('CADE-mini, full', 'cade_pca8_p702/cade_full_8_702'),
          ('CADE-mini, encoder', 'cade_pca8_p702/cade_enc_8_702'), ('CADE-mini, tuned', 'cade_pca8_p702/cade_tuned_8_702'),
          ('CADE-mini, deep', 'cade_pca8_p702/cade_deep_8_702')]
DS = {'EMBER': 'EMBER', 'Drebin': 'Drebin', 'lamda': 'LAMDA'}


def auc(s, y):
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


rows = []
for ds, dname in DS.items():
    fams, tab = {}, {}
    for lab, m in MODELS:
        for sp in range(8):
            vals = []
            for run in range(3):
                d = f'{ds}/pca8/{m}/results/run{run}/split{sp}'
                f = os.path.join(d, 'ood_scores.csv')
                if not os.path.exists(f):
                    continue
                rr = list(csv.DictReader(open(f)))
                s = np.array([float(r['raw_score']) for r in rr]); y = np.array([int(float(r['is_ood'])) for r in rr])
                vals.append(auc(s, y))
                if sp not in fams and os.path.exists(os.path.join(d, 'summary.json')):
                    fams[sp] = str(json.load(open(os.path.join(d, 'summary.json'))).get('ood_family', f'split{sp}'))
            if vals:
                tab[(lab, sp)] = (float(np.mean(vals)), float(np.std(vals)), len(vals))
                rows.append([dname, sp, fams.get(sp, f'split{sp}'), lab, round(np.mean(vals), 4), round(np.std(vals), 4), len(vals)])
    print(f'=== {dname}: per-family ROC-AUC (mean over seeds; * = best in column; n<3 seeds marked with !)')
    print(f"{'model':28s}" + ''.join(f'{fams.get(sp, sp)[:11]:>12s}' for sp in range(8)) + '   mean')
    best = {sp: max((tab[(l, sp)][0] for l, _ in MODELS if (l, sp) in tab), default=None) for sp in range(8)}
    for lab, _ in MODELS:
        cells = []
        for sp in range(8):
            if (lab, sp) not in tab:
                cells.append(f'{"--":>12s}'); continue
            v, _, n = tab[(lab, sp)]
            cells.append(f'{v:11.3f}' + ('*' if v == best[sp] else ('!' if n < 3 else ' ')))
        ms = [tab[(lab, sp)][0] for sp in range(8) if (lab, sp) in tab]
        print(f'{lab:28s}' + ''.join(cells) + (f'  {np.mean(ms):.3f}' if ms else ''))
    print()
with open('PERFAMILY_AUC.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['dataset', 'split', 'family', 'model', 'auc_mean', 'auc_std', 'n_seeds']); w.writerows(rows)
print('Saved PERFAMILY_AUC.csv (row means should match the ROC-AUC column of the main table)')
