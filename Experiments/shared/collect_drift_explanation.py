#!/usr/bin/env python3
"""collect_drift_explanation.py -- run from Experiments2 via SLURM after all drift_analysis_{cade,quantum} jobs are done.
Collects the data for the drift-explanation section (EMBER, Drebin, LAMDA; seed run 0, as the analyses):
  DX_methods.csv    per dataset x model x explanation method (cade, boundary, coin, gradient): for the explained DRIFTING
                    samples, share whose perturbed score falls below the model's 10%-FPR threshold (sufficiency), mean
                    known-family percentile of the perturbed score (CBP), mean k; pooled over the 8 splits
  DX_reupload.csv   quantum models: mean distance after the amplitude encoding (s0) and the cumulative distance after each of
                    the 24 re-uploads, for drifting and known test samples, per dataset (from decomposition.csv)
  DX_mixture.csv    quantum + twins: for each unseen family, mean fidelity of its drifting samples to every known prototype,
                    the novelty residual (part of the state outside the prototype span) and the same for known samples
  DX_case_study/    case_study.txt of the Drebin FakeDoc setting (CADE Table-5 style) for the quantum detector and CADE-minis
Thresholds use the model's own ood_scores.csv of the same split and seed run."""
import os, csv, glob, json, shutil
import numpy as np

DATASETS = {'EMBER': 'EMBER', 'Drebin': 'Drebin', 'lamda': 'LAMDA'}
MODELS = {'quantum_hybrid_v7_p702': ('Quantum (pair + prototype)', 'quantum_pca8/pennylane/quantum_hybrid_v7_p702'),
          'quantum_hybrid_v7_p702_paironly': ('Quantum (pair loss only)', 'ablation_objective/quantum_hybrid_v7_p702_paironly'),
          'classical_fidtwin_c_702': ('Twin, complex', 'twin_pca8_p702/classical_fidtwin_c_702'),
          'classical_fidtwin_p_702': ('Twin, plain', 'twin_pca8_p702/classical_fidtwin_p_702'),
          'classical_fidtwin_r_702': ('Twin, real', 'twin_pca8_p702/classical_fidtwin_r_702'),
          'cade_full_8_702': ('CADE-mini, full', 'cade_pca8_p702/cade_full_8_702'),
          'cade_enc_8_702': ('CADE-mini, encoder', 'cade_pca8_p702/cade_enc_8_702'),
          'cade_tuned_8_702': ('CADE-mini, tuned', 'cade_pca8_p702/cade_tuned_8_702'),
          'cade_deep_8_702': ('CADE-mini, deep', 'cade_pca8_p702/cade_deep_8_702')}
METHODS = ['cade', 'boundary', 'coin', 'gradient']
RUN = 0


def known_scores(ds, folder, split):
    f = f'{ds}/pca8/{folder}/results/run{RUN}/split{split}/ood_scores.csv'
    if not os.path.exists(f):
        return None
    r = list(csv.DictReader(open(f)))
    s = np.array([float(x['raw_score']) for x in r]); y = np.array([int(float(x['is_ood'])) for x in r])
    return np.sort(s[y == 0])


rows_m, rows_r, rows_x, missing = [], [], [], []
for ds, dn in DATASETS.items():
    for tag, (lab, folder) in MODELS.items():
        # ---------------- CADE-protocol explanations
        acc = {m: {'suff': [], 'cbp': [], 'k': []} for m in METHODS}; n_spl = 0
        for sp in range(8):
            f = f'{ds}/pca8/drift_analysis_cade/{tag}/results/split{sp}/explanations.csv'
            ks = known_scores(ds, folder, sp)
            if not os.path.exists(f) or ks is None:
                continue
            n_spl += 1; thr = np.quantile(ks, 0.9)
            for r in csv.DictReader(open(f)):
                if r['is_drift'] != '1':
                    continue
                for m in METHODS:
                    v = r.get(f'd_{m}', '')
                    if v == '':
                        continue
                    d = float(v); acc[m]['suff'].append(d < thr); acc[m]['cbp'].append(np.searchsorted(ks, d) / len(ks))
                    if m == 'cade':
                        acc[m]['k'].append(int(r['k']))
        if n_spl < 8:
            missing.append(f'{dn} {tag} drift_analysis_cade: {n_spl}/8 splits')
        for m in METHODS:
            if acc[m]['suff']:
                rows_m.append([dn, lab, m, n_spl, len(acc[m]['suff']), round(float(np.mean(acc[m]['suff'])), 4),
                               round(float(np.mean(acc[m]['cbp'])), 4), round(float(np.mean(acc[m]['k'])), 2) if acc[m]['k'] else ''])
        # ---------------- re-upload decomposition (quantum models)
        if tag.startswith('quantum'):
            S = {0: [], 1: []}
            for sp in range(8):
                f = f'{ds}/pca8/drift_analysis_quantum/{tag}/results/split{sp}/decomposition.csv'
                if not os.path.exists(f):
                    continue
                for r in csv.DictReader(open(f)):
                    d = [float(r['s0'])] + [float(r[f'Delta_{u}']) for u in range(1, 25)]
                    S[int(r['is_drift'])].append(np.cumsum(d))
            if S[0] and S[1]:
                for g, name in ((1, 'drifting'), (0, 'known')):
                    m_ = np.mean(S[g], 0); q1, q3 = np.percentile(S[g], [25, 75], 0)
                    for u in range(25):
                        rows_r.append([dn, lab, name, u, round(float(m_[u]), 5), round(float(q1[u]), 5), round(float(q3[u]), 5)])
            else:
                missing.append(f'{dn} {tag} decomposition.csv')
        # ---------------- prototype-mixture profile (quantum + twins)
        if tag.startswith('quantum') or tag.startswith('classical'):
            for sp in range(8):
                f = f'{ds}/pca8/drift_analysis_quantum/{tag}/results/split{sp}/mixture.npz'
                sj = f'{ds}/pca8/{folder}/results/run{RUN}/split{sp}/summary.json'
                if not os.path.exists(f):
                    continue
                z = np.load(f, allow_pickle=True)
                fam = str(json.load(open(sj)).get('ood_family', f'split{sp}')) if os.path.exists(sj) else f'split{sp}'
                F, res, ood, names = z['fidelities'], z['residual'], z['is_ood'], [str(n) for n in z['known_names']]
                for g, name in ((1, 'drifting'), (0, 'known')):
                    if (ood == g).sum() == 0:
                        continue
                    mf = F[ood == g].mean(0)
                    rows_x.append([dn, lab, sp, fam, name, round(float(res[ood == g].mean()), 4)] +
                                  ['|'.join(f'{n}:{v:.4f}' for n, v in zip(names, mf))])

with open('DX_methods.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['dataset', 'model', 'method', 'n_splits', 'n_drift_explained', 'sufficiency_fpr10', 'cbp', 'k'])
    w.writerows(rows_m)
with open('DX_reupload.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['dataset', 'model', 'group', 'after_reupload', 'mean_distance', 'q1', 'q3'])
    w.writerows(rows_r)
with open('DX_mixture.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['dataset', 'model', 'split', 'unseen_family', 'group', 'novelty_residual', 'mean_fidelity_to_prototypes'])
    w.writerows(rows_x)
os.makedirs('DX_case_study', exist_ok=True)
for tag in ('quantum_hybrid_v7_p702', 'quantum_hybrid_v7_p702_paironly', 'cade_enc_8_702', 'cade_full_8_702'):
    for f in glob.glob(f'Drebin/pca8/drift_analysis_cade/{tag}/results/split*/case_study.txt'):
        sp = f.split('/')[-2]; shutil.copy(f, f'DX_case_study/{tag}_{sp}.txt')

# ---------------- printed summary
print('=== explanation methods: sufficiency at 10% FPR / CBP (drifting samples, 8 splits pooled)')
for dn in DATASETS.values():
    print(f'-- {dn}')
    for tag, (lab, _) in MODELS.items():
        cells = {r[2]: r for r in rows_m if r[0] == dn and r[1] == lab}
        if cells:
            print(f'  {lab:28s} ' + '  '.join(f'{m}: {cells[m][5]:.3f}/{cells[m][6]:.3f}' if m in cells else f'{m}: --' for m in METHODS)
                  + (f'   k={cells["cade"][7]}' if 'cade' in cells else ''))
print('\n=== re-upload decomposition: mean distance after encoding -> after last re-upload (drifting | known)')
for dn in DATASETS.values():
    for lab in ('Quantum (pair + prototype)', 'Quantum (pair loss only)'):
        d = {(r[2], r[3]): r[4] for r in rows_r if r[0] == dn and r[1] == lab}
        if d:
            print(f'  {dn:7s} {lab:28s} drifting {d[("drifting", 0)]:.3f} -> {d[("drifting", 24)]:.3f}   known {d[("known", 0)]:.3f} -> {d[("known", 24)]:.3f}')
print('\n=== novelty residual of drifting vs known samples, and the known family the drifting family resembles most (quantum full objective)')
for dn in DATASETS.values():
    for r in [r for r in rows_x if r[0] == dn and r[1] == 'Quantum (pair + prototype)' and r[4] == 'drifting']:
        kn = [x for x in rows_x if x[0] == dn and x[1] == r[1] and x[2] == r[2] and x[4] == 'known']
        prof = sorted(((float(v), n) for n, v in (p.split(':') for p in r[6].split('|'))), reverse=True)
        print(f'  {dn:7s} {r[3]:15s} residual drift {r[5]:.3f} vs known {kn[0][5] if kn else float("nan"):.3f}; '
              f'closest prototypes: {prof[0][1]} {prof[0][0]:.2f}, {prof[1][1]} {prof[1][0]:.2f}')
if missing:
    print('\nMISSING:'); [print('  ' + m) for m in missing]
print('\nSaved DX_methods.csv, DX_reupload.csv, DX_mixture.csv, DX_case_study/')
