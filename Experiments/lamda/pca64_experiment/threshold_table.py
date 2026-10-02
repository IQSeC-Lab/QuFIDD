#!/usr/bin/env python3
"""
threshold_table.py -- results at FIXED thresholds on the raw distance and on the MAD score, for every model
===========================================================================================================
Reads the saved per-sample test scores (results*/run*/split*/ood_scores.csv: raw_score, mad_score, is_ood) of
every model in an experiment folder -- nothing is retrained -- and, for each threshold T, flags a sample as
drift if its score > T. Reported per model (mean +- std over tasks) and per family:
  flagged  share of test samples flagged
  P, R     precision and recall of the flagged set;  F1 = their harmonic mean
  FPR      share of KNOWN test samples flagged (false-alarm rate)
  CADE-F1  CADE protocol within the flagged set (rank flagged samples by raw distance, best cut-off)

NOTE ON RAW THRESHOLDS: raw distances have different units per model (quantum: 1 - fidelity in [0, 1];
PCA-L2: Euclidean between unit vectors in [0, 2]; CADE-mini: unbounded Euclidean latent distance). A raw
threshold of 1.0 is therefore strict for one model and lenient for another; compare raw-threshold rows
WITHIN a model. MAD scores are unit-free (robust z-scores), so MAD-threshold rows compare ACROSS models.

RELATIVE MODE (--relative N): instead of fixed values, each task's OBSERVED score range [min, max] (over all
its test samples; no labels used) is divided into N equal steps, giving thresholds
  T_k = min + k * (max - min) / N,   k = 1 .. N-1
e.g. N = 7 on a range [0, 1] -> 0.143, 0.286, 0.429, 0.571, 0.714, 0.857. Done separately for the raw distance
and the MAD score, per task; rows report the mean threshold value and the metrics at each step k. Because every
model is cut at the same RELATIVE positions of its own range, rows are comparable across models (note: a few
extreme scores widen the range and push all thresholds up; see the reported mean T).

Usage (from lamda/):
  python3 threshold_table.py pca64_experiment
  python3 threshold_table.py Full_new_experiments --raw 0.5 1.0 1.5 --mad 0.5 1.0 1.5
  python3 threshold_table.py pca8_experiment --by-family
  python3 threshold_table.py pca64_experiment --relative 7                 # 6 thresholds per score

STANDARD PROTOCOL (per experiment):
  classical models + baselines:  python3 threshold_table.py <exp> --models cade pca --mad 0 2 2.5 3 3.5 --raw --tag classical
      (CADE protocol: flag MAD > T, T in {0, 2, 2.5, 3, 3.5}, T = 0 flags all; rank flagged by raw distance, best cut-off)
  quantum models:                python3 threshold_table.py <exp> --relative 7 --models quantum --scores raw --tag quantum
      (flag RAW distance > T_k = min + k (max - min) / 7 of the observed range, k = 1..6, plus T = 0 (flag all);
       rank flagged by raw distance;
       both F1s: flag-rule F1 (all flagged) and ranked F1 (CADE-F1, best cut-off))
  python3 threshold_table.py Full_new_experiments --relative 7 --models quantum
  python3 threshold_table.py pca64_experiment --models quantum --raw 0.143 0.286 0.429 0.571 0.714 0.857 --mad
      (theoretical raw range [0, 1] / 7 for the quantum model; an empty --mad skips the MAD rows)
Outputs: <experiment>/threshold_table.txt and threshold_table.csv
"""
import os, sys, glob, csv, argparse
import numpy as np


def metrics(score, mad, raw, ood, T):
    # T <= 0 flags EVERY sample (as compare_all.py: T = 0 = threshold-free); otherwise score > T
    flag = (score > T) if T > 0 else np.ones(len(score), bool); act = ood.astype(bool)
    tp, fp, fn = int((flag & act).sum()), int((flag & ~act).sum()), int((~flag & act).sum())
    P = tp / (tp + fp) if tp + fp else 0.0; R = tp / (tp + fn) if tp + fn else 0.0
    F1 = 2 * P * R / (P + R) if P + R else 0.0
    # CADE protocol (Yang et al., USENIX Security 2021): the flagged samples are ranked by raw distance and the
    # analyst inspects the top k; k is the cut-off with the best F1. Effort = k / number of true drifting samples.
    n_ood, n_known = int(ood.sum()), int((~act).sum())
    idx = np.where(flag)[0]; cade = cP = cR = eff = cFPR = 0.0
    if len(idx):
        o = idx[np.argsort(-raw[idx], kind='stable')]
        tpc = np.cumsum(ood[o]); k = np.arange(1, len(o) + 1)
        p, r = tpc / k, tpc / n_ood
        f1 = np.where(p + r > 0, 2 * p * r / (p + r + 1e-12), 0.0); b = int(np.argmax(f1))
        cade, cP, cR, eff = float(f1[b]), float(p[b]), float(r[b]), float(k[b] / n_ood)
        cFPR = float((k[b] - tpc[b]) / n_known)                  # known samples among the k inspected
    return {'flagged': float(flag.mean()), 'P': P, 'R': R, 'F1': F1, 'FNR': 1.0 - R, 'FPR': float(flag[~act].mean()),
            'CADE-P': cP, 'CADE-R': cR, 'CADE-F1': cade, 'CADE-FNR': 1.0 - cR, 'CADE-FPR': cFPR, 'Effort': eff}


def discover(exp):
    """Every results folder with per-task ood_scores.csv -> {model label: {(run, split): (family, raw, mad, ood)}}."""
    models = {}
    for d in sorted(glob.glob(os.path.join(exp, '*', 'results*'))):
        files = sorted(glob.glob(os.path.join(d, 'run*', 'split*', 'ood_scores.csv')))
        if not files:
            continue
        folder, res = os.path.relpath(d, exp).split(os.sep)
        suffix = res[len('results'):].lstrip('_')                   # results -> '', results_l2 -> 'l2', ...
        label = folder + ('' if not suffix else (' (L2)' if suffix == 'l2' else f' ({suffix})'))
        tasks = {}
        for f in files:
            r = int(f.split('run')[-1].split(os.sep)[0]); s = int(f.split('split')[-1].split(os.sep)[0])
            a = np.genfromtxt(f, delimiter=',', names=True)
            fam = '?'
            sj = os.path.join(os.path.dirname(f), 'summary.json')
            if os.path.exists(sj):
                import json
                fam = json.load(open(sj)).get('ood_family', '?')
            tasks[(r, s)] = (fam, a['raw_score'], a['mad_score'], a['is_ood'].astype(int))
        models[label] = tasks
    return models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('experiment')
    ap.add_argument('--raw', type=float, nargs='*', default=[0.5, 1.0, 1.5])
    ap.add_argument('--mad', type=float, nargs='*', default=[0.5, 1.0, 1.5])
    ap.add_argument('--by-family', action='store_true')
    ap.add_argument('--relative', type=int, default=0, help='divide each task\'s observed score range into N steps')
    ap.add_argument('--models', nargs='+', default=None, help='only models whose folder name contains one of these')
    ap.add_argument('--tag', default='', help='suffix for the output file names, e.g. quantum or cade')
    ap.add_argument('--scores', nargs='+', default=['raw', 'MAD'], choices=['raw', 'MAD'],
                    help='relative mode: which scores to threshold (quantum protocol: raw only)')
    a = ap.parse_args()
    exp = os.path.abspath(a.experiment)
    models = discover(exp)
    if a.models:
        models = {m: t for m, t in models.items() if any(x in m for x in a.models)}
    if a.relative:
        return relative(exp, models, a.relative, a.by_family, a.tag, a.scores)
    if not models:
        raise SystemExit(f'No results with ood_scores.csv found under {exp}')
    lines, rows = [], []
    def log(s=''):
        print(s); lines.append(s)
    cols = ['flagged', 'P', 'R', 'F1', 'FNR', 'FPR', 'CADE-P', 'CADE-R', 'CADE-F1', 'CADE-FNR', 'CADE-FPR', 'Effort']
    log('=' * 118); log(f'FIXED-THRESHOLD RESULTS -- {os.path.basename(exp)} (test scores; mean +- std over tasks)'); log('=' * 118)
    log('Raw-distance thresholds: units differ per model -> compare within a model. MAD thresholds: unit-free -> compare across models.')
    for kind, Ts in [('raw', a.raw), ('MAD', a.mad)]:
        if not Ts:
            continue
        log(f'\n### Threshold on the {"RAW DISTANCE" if kind == "raw" else "MAD SCORE"}')
        for T in Ts:
            log(f'\n  {kind} > {T}')
            log(f"    {'model':<30}{'tasks':>6}" + ''.join(f'{c:>16}' for c in cols))
            for m, tasks in models.items():
                res = [metrics(raw if kind == 'raw' else mad, mad, raw, ood, T) for (_, raw, mad, ood) in tasks.values()]
                M = {c: [x[c] for x in res] for c in cols}
                log(f'    {m:<30}{len(res):>6}' + ''.join(f'{np.mean(M[c]):>9.3f}+-{np.std(M[c]):<5.3f}' for c in cols))
                rows.append({'score': kind, 'T': T, 'model': m, 'family': 'ALL', 'tasks': len(res), **{c: np.mean(M[c]) for c in cols}})
                if a.by_family:
                    fams = sorted({v[0] for v in tasks.values()})
                    for fam in fams:
                        rf = [metrics(raw if kind == 'raw' else mad, mad, raw, ood, T) for (f_, raw, mad, ood) in tasks.values() if f_ == fam]
                        Mf = {c: [x[c] for x in rf] for c in cols}
                        log(f'      {fam:<28}{len(rf):>6}' + ''.join(f'{np.mean(Mf[c]):>9.3f}+-{np.std(Mf[c]):<5.3f}' for c in cols))
                        rows.append({'score': kind, 'T': T, 'model': m, 'family': fam, 'tasks': len(rf), **{c: np.mean(Mf[c]) for c in cols}})
    log('\n  Flag rule (all samples above T): flagged share, P, R, F1, FNR = missed drift / all drift, FPR = flagged known / all known.')
    log('  CADE protocol: flagged samples ranked by raw distance, analyst inspects the top k (k = best-F1 cut-off):')
    log('  CADE-P/R/F1/FNR/FPR at that k, Effort = k / number of true drifting samples (CADE\'s normalized effort).')
    base = os.path.join(exp, 'threshold_table' + (f'_{a.tag}' if a.tag else ''))
    open(base + '.txt', 'w').write('\n'.join(lines) + '\n')
    with open(base + '.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['score', 'T', 'model', 'family', 'tasks'] + cols); w.writeheader(); w.writerows(rows)
    print(f'\nSaved: {base}.txt, {base}.csv')


def relative(exp, models, N, by_family, tag='', scores=('raw', 'MAD')):
    lines, rows = [], []
    def log(s=''):
        print(s); lines.append(s)
    cols = ['T', 'flagged', 'P', 'R', 'F1', 'FNR', 'FPR', 'CADE-P', 'CADE-R', 'CADE-F1', 'CADE-FNR', 'CADE-FPR', 'Effort']
    log('=' * 124); log(f'RELATIVE THRESHOLDS -- {os.path.basename(exp)}: each task\'s observed score range split into {N} steps'
                        f' (T_k = min + k (max - min) / {N}); mean +- std over tasks'); log('=' * 124)
    for kind in scores:
        log(f'\n### Threshold on the {"RAW DISTANCE" if kind == "raw" else "MAD SCORE"}')
        for m, tasks in models.items():
            log(f'\n  {m}  ({len(tasks)} tasks)')
            log(f"    {'k / ' + str(N):<10}" + ''.join(f'{c:>16}' for c in cols))
            for k in range(0, N):                     # k = 0: T = 0, every sample flagged (threshold-free)
                res = []
                per_fam = {}
                for (fam, raw, mad, ood) in tasks.values():
                    sc = raw if kind == 'raw' else mad
                    lo, hi = float(sc.min()), float(sc.max()); T = 0.0 if k == 0 else lo + k * (hi - lo) / N
                    r = metrics(sc, mad, raw, ood, T); r['T'] = T; res.append(r); per_fam.setdefault(fam, []).append(r)
                M = {c: [x[c] for x in res] for c in cols}
                log(f'    {k:<10}' + ''.join(f'{np.mean(M[c]):>9.3f}+-{np.std(M[c]):<5.3f}' for c in cols))
                rows.append({'score': kind, 'k': k, 'N': N, 'model': m, 'family': 'ALL', 'tasks': len(res), **{c: np.mean(M[c]) for c in cols}})
                if by_family:
                    for fam in sorted(per_fam):
                        Mf = {c: [x[c] for x in per_fam[fam]] for c in cols}
                        log(f'      {fam:<8}' + ''.join(f'{np.mean(Mf[c]):>9.3f}+-{np.std(Mf[c]):<5.3f}' for c in cols))
                        rows.append({'score': kind, 'k': k, 'N': N, 'model': m, 'family': fam, 'tasks': len(per_fam[fam]),
                                     **{c: np.mean(Mf[c]) for c in cols}})
    log('\n  T = threshold value (mean over tasks). Flag rule (all samples above T): flagged share, P, R, F1, FNR, FPR.')
    log('  CADE protocol: flagged samples ranked by raw distance, analyst inspects the top k (k = best-F1 cut-off):')
    log('  CADE-P/R/F1/FNR/FPR at that k, Effort = k / number of true drifting samples.')
    base = os.path.join(exp, f'threshold_relative_{N}' + (f'_{tag}' if tag else ''))
    open(base + '.txt', 'w').write('\n'.join(lines) + '\n')
    with open(base + '.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['score', 'k', 'N', 'model', 'family', 'tasks'] + cols); w.writeheader(); w.writerows(rows)
    print(f'\nSaved: {base}.txt, .csv')


if __name__ == '__main__':
    main()
