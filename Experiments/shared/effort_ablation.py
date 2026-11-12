#!/usr/bin/env python3
"""effort_ablation.py -- run from Experiments2 (via run_effort_ablation.slurm)
CADE inspection effort at the stopping point of the ranked F1, per task (run, split), mean over tasks.
  ranked F1: sort test samples by raw_score (high = more drifting), F1 at every cut-off k, keep the best k;
  effort   : k / number of drifting samples (inspected samples per drifting sample).
Reports the ablation models plus four reference models already in Table I, so the computation can be checked
against the effort values in the table before the new numbers are used."""
import os, glob, csv
import numpy as np

REFERENCE = {  # in Table I -> effort there (EMBER, Drebin, LAMDA)
    'quantum_pca8/pennylane/quantum_hybrid_v7_p702': (1.26, 1.07, 1.69),
    'twin_pca8_p702/classical_fidtwin_c_702': (1.29, 1.22, 1.62),
    'cade_pca8_p702/cade_enc_8_702': (1.71, 1.40, 1.78),
}


def task_effort(f):
    rows = list(csv.DictReader(open(f)))
    s = np.array([float(r['raw_score']) for r in rows]); y = np.array([int(float(r['is_ood'])) for r in rows])
    order = np.argsort(-s, kind='stable'); y = y[order]
    tp = np.cumsum(y); k = np.arange(1, len(y) + 1); P = y.sum()
    f1 = 2 * tp / (k + P)                       # 2TP / (TP+FP + TP+FN)
    best = int(np.argmax(f1))
    return k[best] / P, f1[best], (k[best] - tp[best]) / max(1, len(y) - P)


def model_effort(folder):
    files = glob.glob(os.path.join(folder, 'results', 'run*', 'split*', 'ood_scores.csv'))
    if not files:
        return None
    e = np.array([task_effort(f) for f in files])
    return len(files), e[:, 0].mean(), e[:, 1].mean(), e[:, 2].mean()


out = []
for di, ds in enumerate(['EMBER', 'Drebin', 'lamda']):
    print(f'=== {ds}')
    folders = [(m, f'{ds}/pca8/{m}', REFERENCE[m][di]) for m in REFERENCE]
    folders += [(os.path.basename(a.rstrip('/')), a, None) for a in sorted(glob.glob(f'{ds}/pca8/ablation_objective/*/'))]
    for name, folder, ref in folders:
        r = model_effort(folder)
        if r is None:
            print(f'  {name:55s} no ood_scores.csv found'); continue
        n, eff, f1, fpr = r
        chk = f'   (Table I: {ref:.2f})' if ref is not None else ''
        print(f'  {name:55s} n={n:2d}  effort {eff:.2f}  rankedF1 {f1:.3f}  FPR {fpr:.3f}{chk}')
        out.append([ds, name, n, f'{eff:.3f}', f'{f1:.4f}', f'{fpr:.4f}'])

with open('EFFORT_ABLATION.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['dataset', 'model', 'n_tasks', 'effort', 'rankedF1', 'FPR']); w.writerows(out)
print('\nSaved EFFORT_ABLATION.csv')
