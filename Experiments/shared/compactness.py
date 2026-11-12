#!/usr/bin/env python3
"""compactness.py -- run from Experiments2 via SLURM. Effect of the prototype loss on the fidelity geometry.
For the quantum detector and the three twins, under the CADE objective (ablation_objective/*_paironly) and the full
objective (main folders), over all 24 tasks per dataset, from the saved states (states.npz):
  own_F    mean fidelity of KNOWN TEST samples to the prototype of their own family   (compactness, higher = tighter)
  train_F  the same for training samples
  proto_F  mean fidelity between prototypes of different known families              (overlap, lower = more orthogonal)
  drift_F  mean fidelity of DRIFTING samples to their nearest prototype                (lower = drift kept further away)
  known_F  mean fidelity of KNOWN test samples to their nearest prototype (the control for drift_F)
  gap      known_F - drift_F
Prototype = leading eigenvector of the family's training density matrix (as in the detector).
Paired Wilcoxon (CADE objective vs. full objective) over the tasks for own_F, proto_F and drift_F."""
import os, glob, csv
import numpy as np
from scipy.stats import wilcoxon

MODELS = {'Quantum': ('quantum_pca8/pennylane/quantum_hybrid_v7_p702', 'ablation_objective/quantum_hybrid_v7_p702_paironly'),
          'Twin, complex': ('twin_pca8_p702/classical_fidtwin_c_702', 'ablation_objective/classical_fidtwin_c_702_paironly'),
          'Twin, plain': ('twin_pca8_p702/classical_fidtwin_p_702', 'ablation_objective/classical_fidtwin_p_702_paironly'),
          'Twin, real': ('twin_pca8_p702/classical_fidtwin_r_702', 'ablation_objective/classical_fidtwin_r_702_paironly')}
checked = set()


def task_stats(d):
    f = os.path.join(d, 'states.npz')
    if not os.path.exists(f):
        return None
    z = np.load(f)
    Str = z[[k for k in z.files if k.startswith('states_train')][0]]; Ste = z[[k for k in z.files if k.startswith('states_test')][0]]
    ytr, yte = z['y_train'], z['y_test']
    Str = Str / np.linalg.norm(Str, axis=1, keepdims=True); Ste = Ste / np.linalg.norm(Ste, axis=1, keepdims=True)
    labs = np.unique(ytr); P = []
    for c in labs:
        S = Str[ytr == c]; w, V = np.linalg.eigh(S.T @ S.conj() / len(S)); P.append(V[:, -1])
    P = np.array(P)
    Fte = np.abs(Ste @ P.conj().T) ** 2; Ftr = np.abs(Str @ P.conj().T) ** 2
    # ground truth for drifting samples: is_ood of ood_scores.csv (y_test keeps the ORIGINAL family ids, while y_train
    # may be renumbered 0..N-1 in sorted order of the original known ids)
    rows = list(csv.DictReader(open(os.path.join(d, 'ood_scores.csv'))))
    s = np.array([float(r['raw_score']) for r in rows]); is_ood = np.array([int(float(r['is_ood'])) for r in rows])
    if len(s) != len(Fte):
        print(f'  !! {d}: {len(s)} scores vs {len(Fte)} test states -- skipped'); return None
    known = is_ood == 0
    known_te = set(int(v) for v in np.unique(yte[known])); held = set(int(v) for v in np.unique(yte[~known]))
    if known_te <= set(int(c) for c in labs) and not (held & set(int(c) for c in labs)):
        idx = {int(c): i for i, c in enumerate(labs)}                              # y_train keeps the original ids
    else:  # y_train renumbered 0..N-1: original ids 0..N with the held-out id removed, in sorted order
        orig = [c for c in range(len(labs) + len(held)) if c not in held]
        if len(orig) != len(labs) or not known_te <= set(orig):
            print(f'  !! {d}: cannot map test families {sorted(known_te)} (held out {sorted(held)}) to {len(labs)} prototypes -- skipped')
            return None
        idx = {o: i for i, o in enumerate(orig)}
    own_te = np.array([Fte[i, idx[int(c)]] for i, c in enumerate(yte) if known[i]])
    own_tr = np.array([Ftr[i, i_] for i, i_ in enumerate(np.searchsorted(labs, ytr))])
    acc = np.mean([np.argmax(Fte[i]) == idx[int(c)] for i, c in enumerate(yte) if known[i]])  # nearest prototype = own family
    G = np.abs(P @ P.conj().T) ** 2; off = G[~np.eye(len(P), dtype=bool)]
    key = os.path.dirname(os.path.dirname(os.path.dirname(d)))
    if key not in checked:  # once per model folder: scores reproduced, and the family mapping is plausible
        checked.add(key)
        err = np.max(np.abs(1 - Fte.max(1) - s))
        print(f'  CHECK {key}: max |(1-F) - ood_scores.csv| = {err:.1e}' + ('' if err < 1e-3 else '  !! MISMATCH')
              + f';  known test samples nearest to their own prototype: {100 * acc:.0f}% (chance {100 / len(labs):.0f}%)')
    return own_te.mean(), own_tr.mean(), off.mean(), Fte[~known].max(1).mean(), Fte[known].max(1).mean()


def collect(folder):
    out = {}
    for d in glob.glob(os.path.join(folder, 'results', 'run*', 'split*')):
        t = task_stats(d)
        if t is not None:
            out[tuple(d.split(os.sep)[-2:])] = t
    return out


NAMES = ['own_F', 'train_F', 'proto_F', 'drift_F', 'known_F', 'gap']
for ds in ['EMBER', 'Drebin', 'lamda']:
    print(f'=== {ds}')
    print(f"  {'model':15s} {'n':>3s}  " + '  '.join(f'{n:>17s}' for n in NAMES) + '   p(own_F) p(proto_F) p(drift_F) p(known_F) p(gap)')
    for lab, (full, cade) in MODELS.items():
        Fu, Ca = collect(f'{ds}/pca8/{full}'), collect(f'{ds}/pca8/{cade}')
        keys = sorted(set(Fu) & set(Ca))
        if not keys:
            print(f'  {lab:15s} no paired tasks with states.npz ({len(Fu)} full, {len(Ca)} CADE-objective)'); continue
        a = np.array([Ca[k] for k in keys]); b = np.array([Fu[k] for k in keys])
        a = np.c_[a, a[:, 4] - a[:, 3]]; b = np.c_[b, b[:, 4] - b[:, 3]]   # gap = known_F - drift_F (nearest prototype)
        cells = '  '.join(f'{a[:, j].mean():.3f} -> {b[:, j].mean():.3f}' for j in range(6))
        ps = [wilcoxon(b[:, j], a[:, j]).pvalue for j in (0, 2, 3, 4, 5)]
        print(f'  {lab:15s} {len(keys):3d}  {cells}   ' + '  '.join(f'{p:8.2g}' for p in ps))
print('cells: CADE objective -> full objective (mean over tasks). own_F up = known families tighter around their prototypes;'
      ' proto_F down = prototypes more orthogonal; drift_F down = drifting samples further from every prototype; known_F = known test samples to their NEAREST prototype; gap = known_F - drift_F (larger = drift separated better).')
