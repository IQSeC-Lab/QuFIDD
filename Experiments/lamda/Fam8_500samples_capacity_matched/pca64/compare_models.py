#!/usr/bin/env python3
"""
compare_models.py -- side-by-side comparison of all models in one experiment folder.

Reads results/run*/split*/summary.json (and results_l2 for pca_only) written by the run_*.py scripts, prints one
table per metric group (mean +- std over the 24 tasks, per family and overall), and paired Wilcoxon signed-rank tests
of every model against a reference model (default: quantum_hybrid_v7), paired on (seed, held-out family).
Saves comparison.txt and comparison.csv next to the models.

Usage (from an experiment folder such as Fam8_500samples_capacity_matched/pca8/):
  python3 compare_models.py
  python3 compare_models.py --ref quantum_hybrid_v7 --extra ../../classical_full_capacity_models/Fam8_500samples
"""
import os, sys, json, glob, argparse, csv
import numpy as np

MODELS = [  # (folder, results subfolder, short label)
    ('pca_only', 'results_l2', 'PCA only (L2), 0p'),
    ('cade_full_8', 'results', 'CADE-mini full, 207p'),
    ('cade_deep_8_384', 'results', 'CADE-mini deep, 391p'),
    ('quantum_hybrid_v7', 'results', 'Quantum hybrid v7, 384p'),
    ('classical_fidtwin_c', 'results', 'Fid twin complex, 379p'),
    ('classical_fidtwin_p', 'results', 'Fid twin plain, 379p'),
    ('classical_fidtwin_r', 'results', 'Fid twin real, 383p'),
    ('quantum_hybrid_v7_64', 'results', 'Quantum hybrid v7-64, 474p'),
    ('classical_fidtwin_c64', 'results', 'Fid twin-64 complex, 642p'),
    ('classical_fidtwin_r64', 'results', 'Fid twin-64 real, 450p'),
    ('cade_original', 'results', 'CADE original (full feat.)'),
    ('fidelity_full_c', 'results', 'Fidelity full complex'),
    ('fidelity_full_r', 'results', 'Fidelity full real'),
]
METRICS = [('ROC_raw', lambda e: e['threshold_free']['ROC_AUC']),
           ('ROC_mad', lambda e: e['ROC_AUC_mad']),
           ('PR_raw', lambda e: e['threshold_free']['PR_AUC']),
           ('TF_F1', lambda e: e['threshold_free']['F1']),
           ('F1@T3.5', lambda e: [x for x in e['sweep'] if x['T_MAD'] == 3.5][0]['F1']),
           ('FPR@T3.5', lambda e: [x for x in e['sweep'] if x['T_MAD'] == 3.5][0]['FPR']),
           ('FNR@T3.5', lambda e: [x for x in e['sweep'] if x['T_MAD'] == 3.5][0]['FNR'])]


def load(root, folder, sub):
    out = {}
    for p in glob.glob(os.path.join(root, folder, sub, 'run*', 'split*', 'summary.json')):
        d = json.load(open(p))
        key = (d['run_id'], d['split_id'])
        out[key] = {'family': d['ood_family'], 'params': d.get('n_trainable_params', 0),
                    **{m: f(d['evaluation']) for m, f in METRICS}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.', help='experiment folder holding the model folders')
    ap.add_argument('--extra', nargs='*', default=[], help='more folders to search for models (e.g. full-capacity)')
    ap.add_argument('--ref', default='quantum_hybrid_v7', help='reference model for the paired tests')
    a = ap.parse_args()
    roots = [a.root] + a.extra
    found = {}
    for folder, sub, label in MODELS:
        for r in roots:
            res = load(r, folder, sub)
            if res:
                found[folder] = (label, res, r)
                break
    if not found:
        sys.exit('no results found')
    lines = []
    def log(s=''):
        print(s); lines.append(s)
    log('=' * 118)
    log(f'MODEL COMPARISON  ({os.path.abspath(a.root)})')
    log('=' * 118)
    log(f'{"model":<30}{"tasks":>6}{"params":>9}' + ''.join(f'{m:>15}' for m, _ in METRICS))
    rows = []
    for folder, (label, res, r) in found.items():
        v = list(res.values())
        log(f'{label:<30}{len(v):>6}{v[0]["params"]:>9}' +
            ''.join(f'{np.mean([x[m] for x in v]):>8.3f}+-{np.std([x[m] for x in v]):<5.3f}' for m, _ in METRICS))
        rows.append({'model': folder, 'label': label, 'tasks': len(v), 'params': v[0]['params'],
                     **{f'{m}_mean': np.mean([x[m] for x in v]) for m, _ in METRICS},
                     **{f'{m}_std': np.std([x[m] for x in v]) for m, _ in METRICS}})
    fams = []
    for _, res, _ in found.values():
        for x in res.values():
            if x['family'] not in fams:
                fams.append(x['family'])
    for m in ('ROC_raw', 'ROC_mad', 'F1@T3.5'):
        log(f'\n  Per family: {m} (mean over seeds)')
        log(f'  {"model":<30}' + ''.join(f'{f[:11]:>12}' for f in fams))
        for folder, (label, res, _) in found.items():
            log(f'  {label:<30}' + ''.join(
                f'{np.mean([x[m] for x in res.values() if x["family"] == f]):>12.3f}'
                if any(x['family'] == f for x in res.values()) else f'{"-":>12}' for f in fams))
    if a.ref not in found:                      # e.g. the PCA-64 folder: fall back to its quantum model
        alt = [f for f in found if f.startswith('quantum_hybrid')]
        if alt:
            a.ref = alt[0]
    if a.ref in found:
        try:
            from scipy.stats import wilcoxon
        except ImportError:
            wilcoxon = None
        ref = found[a.ref][1]
        log(f'\n  Paired Wilcoxon vs {found[a.ref][0]} (pairs = same seed and held-out family; diff = model - ref)')
        log(f'  {"model":<30}' + ''.join(f'{m + " diff (p)":>22}' for m in ('ROC_raw', 'ROC_mad', 'F1@T3.5')))
        for folder, (label, res, _) in found.items():
            if folder == a.ref:
                continue
            keys = sorted(set(res) & set(ref))
            cells = []
            for m in ('ROC_raw', 'ROC_mad', 'F1@T3.5'):
                d = np.array([res[k][m] - ref[k][m] for k in keys])
                p = float('nan')
                if wilcoxon is not None and len(d) > 0 and np.any(d != 0):
                    p = wilcoxon(d).pvalue
                cells.append(f'{d.mean():>+12.3f} ({p:.3g})' if len(d) else f'{"-":>22}')
            log(f'  {label:<30}' + ''.join(f'{c:>22}' for c in cells) + f'   n={len(keys)}')
        log('  Note: pca_only is deterministic, so its 3 seeds repeat the same 8 values.')
    out = os.path.abspath(a.root)
    open(os.path.join(out, 'comparison.txt'), 'w').write('\n'.join(lines) + '\n')
    with open(os.path.join(out, 'comparison.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f'\nSaved: {out}/comparison.txt, comparison.csv')


if __name__ == '__main__':
    main()
