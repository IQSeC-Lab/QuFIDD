#!/usr/bin/env python3
"""gap_all.py -- run from Experiments2 (via run_gap_all.slurm).
Mean drift score (distance to the nearest prototype/centroid) of known and drifting test samples,
for every model with the pairwise loss alone and with the added prototype loss.
Per seed: mean over the 8 splits; reported: mean over the 3 seeds (same as the old 24-task mean).
For the fidelity models, 1 - score is the fidelity shown in Fig. 4 (sanity check: QuFiDD EMBER pair known ~0.914).
For the CMs the distance is unbounded, so we also print drifting/known (a scale-free separation).
Output: GAP_ALL.csv"""
import os, glob, csv
from collections import defaultdict
import numpy as np

DATASETS = ['EMBER', 'Drebin', 'lamda']
PAIRS = [  # (name, pair-only folder, pair+proto folder)
    ('QuFiDD', 'ablation_objective/quantum_hybrid_v7_p702_paironly', 'quantum_pca8/pennylane/quantum_hybrid_v7_p702'),
    ('QiCD^c', 'ablation_objective/classical_fidtwin_c_702_paironly', 'twin_pca8_p702/classical_fidtwin_c_702'),
    ('QiCD^p', 'ablation_objective/classical_fidtwin_p_702_paironly', 'twin_pca8_p702/classical_fidtwin_p_702'),
    ('QiCD^r', 'ablation_objective/classical_fidtwin_r_702_paironly', 'twin_pca8_p702/classical_fidtwin_r_702'),
    ('CM^f', 'cade_pca8_p702/cade_full_8_702', 'cade_pca8_p702_proto/cade_full_8_702_proto/tau_10'),
    ('CM^e', 'cade_pca8_p702/cade_enc_8_702', 'cade_pca8_p702_proto/cade_enc_8_702_proto/tau_1'),
    ('CM^t', 'cade_pca8_p702/cade_tuned_8_702', 'cade_pca8_p702_proto/cade_tuned_8_702_proto/tau_10'),
    ('CM^d', 'cade_pca8_p702/cade_deep_8_702', 'cade_pca8_p702_proto/cade_deep_8_702_proto/tau_1'),
]


def stats(ds, fold):
    files = sorted(glob.glob(os.path.join(ds, 'pca8', fold, 'results', 'run*', 'split*', 'ood_scores.csv')))
    if not files:
        return None
    by_run = defaultdict(list)
    for f in files:
        rows = list(csv.DictReader(open(f)))
        s = np.array([float(r['raw_score']) for r in rows]); y = np.array([int(float(r['is_ood'])) for r in rows])
        by_run[os.path.basename(os.path.dirname(os.path.dirname(f)))].append(
            (s[y == 0].mean(), s[y == 1].mean(), np.median(s[y == 0]), np.median(s[y == 1]), s.min(), s.max()))
    per_run = np.array([np.mean(by_run[r], axis=0) for r in sorted(by_run)])
    return per_run.mean(axis=0), len(by_run), len(files)


out = []
for ds in DATASETS:
    print(f'=== {ds}')
    print(f'  {"model":8s} {"loss":10s} {"known":>8s} {"drift":>8s} {"drift/known":>11s} {"med known":>9s} {"med drift":>9s} {"min":>8s} {"max":>8s}  (1-known, 1-drift)  runs/files')
    for name, fp, fpp in PAIRS:
        for loss, fold in (('pair', fp), ('pair+proto', fpp)):
            r = stats(ds, fold)
            if r is None:
                print(f'  {name:8s} {loss:10s} no ood_scores.csv in {fold}'); continue
            (k, d, mk, md, lo, hi), nr, nf = r
            fid = f'({1-k:.3f}, {1-d:.3f})' if not name.startswith('CM') else ''
            print(f'  {name:8s} {loss:10s} {k:8.4f} {d:8.4f} {d/k:11.2f} {mk:9.4f} {md:9.4f} {lo:8.3f} {hi:8.3f}  {fid:18s}  {nr}/{nf}')
            out.append([ds, name, loss, fold, nr, nf, f'{k:.5f}', f'{d:.5f}', f'{d/k:.4f}', f'{mk:.5f}', f'{md:.5f}'])

with open('GAP_ALL.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['dataset', 'model', 'loss', 'folder', 'n_runs', 'n_files', 'known_mean', 'drift_mean', 'drift_over_known', 'known_median', 'drift_median'])
    w.writerows(out)
print('Saved GAP_ALL.csv')
