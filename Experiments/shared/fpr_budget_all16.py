#!/usr/bin/env python3
"""fpr_budget_all16.py -- run from Experiments2 (via run_fpr_budget_all16.slurm).
Ranked F1 under a false-positive budget for all 16 models (both losses for every model type), per dataset.
For each task (run, split): rank test samples by raw_score (high = drifting); at every cut-off compute the FPR on the
known families; the ranked F1 at budget b is the best F1 among cut-offs whose FPR <= b (0 if none). Mean/std over tasks.
Also prints, for the models already in FPR_BUDGET_ALL.csv, the stored value next to the recomputed one (check).
Output: FPR_BUDGET_ALL16.csv"""
import os, glob, csv
import numpy as np

BUDGETS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
DATASETS = ['EMBER', 'Drebin', 'lamda']
# (group, label, folder under <ds>/pca8)
MODELS = [
    ('Quantum', 'Quantum (pair)', 'ablation_objective/quantum_hybrid_v7_p702_paironly'),
    ('Quantum', 'Quantum (pair + prototype)', 'quantum_pca8/pennylane/quantum_hybrid_v7_p702'),
    ('Twin (pair)', 'Twin, complex (pair)', 'ablation_objective/classical_fidtwin_c_702_paironly'),
    ('Twin (pair)', 'Twin, plain (pair)', 'ablation_objective/classical_fidtwin_p_702_paironly'),
    ('Twin (pair)', 'Twin, real (pair)', 'ablation_objective/classical_fidtwin_r_702_paironly'),
    ('Twin (pair + prototype)', 'Twin, complex (pair + prototype)', 'twin_pca8_p702/classical_fidtwin_c_702'),
    ('Twin (pair + prototype)', 'Twin, plain (pair + prototype)', 'twin_pca8_p702/classical_fidtwin_p_702'),
    ('Twin (pair + prototype)', 'Twin, real (pair + prototype)', 'twin_pca8_p702/classical_fidtwin_r_702'),
    ('CADE-mini (pair)', 'CADE-mini, full (recon. + pair)', 'cade_pca8_p702/cade_full_8_702'),
    ('CADE-mini (pair)', 'CADE-mini, encoder (pair)', 'cade_pca8_p702/cade_enc_8_702'),
    ('CADE-mini (pair)', 'CADE-mini, tuned (pair)', 'cade_pca8_p702/cade_tuned_8_702'),
    ('CADE-mini (pair)', 'CADE-mini, deep (pair)', 'cade_pca8_p702/cade_deep_8_702'),
    ('CADE-mini (pair + prototype)', 'CADE-mini, full (recon. + pair + prototype)', 'cade_pca8_p702_proto/cade_full_8_702_proto/tau_10'),
    ('CADE-mini (pair + prototype)', 'CADE-mini, encoder (pair + prototype)', 'cade_pca8_p702_proto/cade_enc_8_702_proto/tau_1'),
    ('CADE-mini (pair + prototype)', 'CADE-mini, tuned (pair + prototype)', 'cade_pca8_p702_proto/cade_tuned_8_702_proto/tau_10'),
    ('CADE-mini (pair + prototype)', 'CADE-mini, deep (pair + prototype)', 'cade_pca8_p702_proto/cade_deep_8_702_proto/tau_1'),
]


def task_curve(f):
    rows = list(csv.DictReader(open(f)))
    s = np.array([float(r['raw_score']) for r in rows]); y = np.array([int(float(r['is_ood'])) for r in rows])
    order = np.argsort(-s, kind='stable'); y = y[order]
    P = y.sum(); N = len(y) - P
    tp = np.cumsum(y); k = np.arange(1, len(y) + 1); fp = k - tp
    f1 = 2 * tp / (k + P); fpr = fp / N
    out = []
    for b in BUDGETS:
        ok = fpr <= b / 100 + 1e-12
        out.append(f1[ok].max() if ok.any() else 0.0)
    return np.array(out)


def load_stored():
    st = {}
    if os.path.exists('FPR_BUDGET_ALL.csv'):
        for r in csv.DictReader(open('FPR_BUDGET_ALL.csv')):
            st[(r['dataset'], r['model'], int(float(r['fpr_budget_pct'])))] = float(r['rankedF1_mean'])
    return st


stored = load_stored()
rows = []
for ds in DATASETS:
    print(f'=== {ds}')
    for grp, lab, fold in MODELS:
        files = sorted(glob.glob(os.path.join(ds, 'pca8', fold, 'results', 'run*', 'split*', 'ood_scores.csv')))
        if not files:
            print(f'  {lab:45s} no ood_scores.csv in {fold}'); continue
        C = np.stack([task_curve(f) for f in files]) * 100
        m, sd = C.mean(0), C.std(0)
        chk = []
        for i, b in enumerate(BUDGETS):
            key = next((k for k in stored if k[0] == ds and k[2] == b and k[1].replace(' loss only', '') == lab), None)
            if key and b in (10, 30):
                chk.append(f'@{b}: {m[i]:.1f}/{stored[key]:.1f}')
        print(f'  {lab:45s} n={len(files):2d}  ' + ' '.join(f'{v:5.1f}' for v in m[:6]) + ('   check ' + ', '.join(chk) if chk else ''))
        for i, b in enumerate(BUDGETS):
            rows.append([ds, grp, lab, len(files), b, f'{m[i]:.2f}', f'{sd[i]:.2f}'])

with open('FPR_BUDGET_ALL16.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['dataset', 'group', 'model', 'n_tasks', 'fpr_budget_pct', 'rankedF1_mean', 'rankedF1_std'])
    w.writerows(rows)
print('\ncolumns above: ranked F1 (%) at budgets 10..60;  check = recomputed/stored in FPR_BUDGET_ALL.csv')
print('Saved FPR_BUDGET_ALL16.csv')
