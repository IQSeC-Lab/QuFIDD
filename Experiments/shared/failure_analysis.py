#!/usr/bin/env python3
"""failure_analysis.py -- run from Experiments2 via SLURM. The two failure modes of the quantum detector (full objective)
in its weakest settings, from states.npz + ood_scores.csv, pooled over the three seeds:
  (1) absorbed part : drifting samples that score below the median known test sample, and the known prototype
                      (family) they sit nearest to;
  (2) false positives: known test samples ranked above the ranked-F1 stopping point, by family, against each family's
                      share of the known test samples (enrichment > 1 = over-represented among the false positives).
Usage: python3 failure_analysis.py [--model quantum_pca8/pennylane/quantum_hybrid_v7_p702]"""
import os, csv, json, argparse
import numpy as np
from collections import Counter

SETTINGS = [('EMBER', 'xtrat'), ('Drebin', 'FakeInstaller'), ('lamda', 'plankton'), ('lamda', 'fakeinst')]
ap = argparse.ArgumentParser(); ap.add_argument('--model', default='quantum_pca8/pennylane/quantum_hybrid_v7_p702')
a = ap.parse_args()


def names_from_summary(sj):
    """try to find an id -> family-name list in summary.json"""
    for k, v in sj.items():
        if isinstance(v, list) and len(v) >= 7 and all(isinstance(x, str) for x in v) and 'famil' in k.lower():
            return k, v
    return None, None


for ds, fam in SETTINGS:
    base = f'{ds}/pca8/{a.model}/results'
    split = None
    for sp in range(8):
        f = f'{base}/run0/split{sp}/summary.json'
        if os.path.exists(f) and str(json.load(open(f)).get('ood_family', '')).lower() == fam.lower():
            split = sp; break
    if split is None:
        print(f'=== {ds} {fam}: setting not found'); continue
    absorbed_near, n_drift, n_abs = Counter(), 0, 0
    fp_fam, known_fam, n_fp, n_known = Counter(), Counter(), 0, 0
    names = None
    for run in range(3):
        d = f'{base}/run{run}/split{split}'
        z = np.load(f'{d}/states.npz'); sj = json.load(open(f'{d}/summary.json'))
        if run == 0:
            key, names = names_from_summary(sj)
            print(f'=== {ds}, unseen {fam} (split {split}); summary.json keys: {sorted(sj.keys())}' + (f'; family names from "{key}"' if key else ''))
        Str = z[[k for k in z.files if k.startswith('states_train')][0]]; Ste = z[[k for k in z.files if k.startswith('states_test')][0]]
        ytr, yte = z['y_train'], z['y_test']
        Str = Str / np.linalg.norm(Str, axis=1, keepdims=True); Ste = Ste / np.linalg.norm(Ste, axis=1, keepdims=True)
        labs = np.unique(ytr); P = []
        for c in labs:
            S = Str[ytr == c]; w, V = np.linalg.eigh(S.T @ S.conj() / len(S)); P.append(V[:, -1])
        P = np.array(P); F = np.abs(Ste @ P.conj().T) ** 2
        rows = list(csv.DictReader(open(f'{d}/ood_scores.csv')))
        s = np.array([float(r['raw_score']) for r in rows]); ood = np.array([int(float(r['is_ood'])) for r in rows])
        assert len(s) == len(F) and np.max(np.abs(1 - F.max(1) - s)) < 1e-3, 'rebuilt scores do not match ood_scores.csv'
        known = ood == 0; held = set(int(v) for v in np.unique(yte[~known]))
        if set(int(v) for v in np.unique(yte[known])) <= set(int(c) for c in labs) and not (held & set(int(c) for c in labs)):
            proto_orig = [int(c) for c in labs]
        else:
            proto_orig = [c for c in range(len(labs) + len(held)) if c not in held]
        # (1) absorbed part
        med = np.median(s[known]); dr = np.where(~known)[0]; ab = dr[s[dr] < med]
        n_drift += len(dr); n_abs += len(ab)
        absorbed_near.update(proto_orig[i] for i in F[ab].argmax(1))
        # (2) false positives above the ranked-F1 stopping point
        o = np.argsort(-s, kind='stable'); y = ood[o]; tp = np.cumsum(y); k = np.arange(1, len(y) + 1)
        f1 = 2 * tp / (k + ood.sum()); stop = int(np.argmax(f1)) + 1
        fp = o[:stop][~ood[o[:stop]].astype(bool)]
        n_fp += len(fp); n_known += known.sum()
        fp_fam.update(int(v) for v in yte[fp]); known_fam.update(int(v) for v in yte[known])
    nm = (lambda i: names[i] if names and i < len(names) else f'family {i}')
    print(f'  (1) absorbed part: {n_abs}/{n_drift} drifting samples ({100 * n_abs / n_drift:.0f}%) score below the median known sample; nearest prototype:')
    for c, n in absorbed_near.most_common():
        print(f'        {nm(c):16s} {n:6d}  ({100 * n / max(n_abs, 1):.0f}% of the absorbed part)')
    print(f'  (2) false positives at the ranked-F1 stopping point: {n_fp} of {n_known} known test samples ({100 * n_fp / n_known:.1f}%); by family:')
    for c, n in fp_fam.most_common():
        share_fp, share_known = n / n_fp, known_fam[c] / n_known
        print(f'        {nm(c):16s} {n:6d}  {100 * share_fp:5.1f}% of FPs vs {100 * share_known:5.1f}% of known test   enrichment {share_fp / share_known:4.2f}')
    print()
