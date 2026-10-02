#!/usr/bin/env python3
"""
evaluate.py -- capacity_matched (top level)   thresholded detection results for every model of an experiment
======================================================================================================================
Reads each model's saved per-sample TEST scores (results*/run*/split*/ood_scores.csv: raw_score, mad_score, is_ood);
nothing is retrained. For every threshold T a sample is FLAGGED if its score > T (T = 0 flags EVERY sample).
Two result modes per threshold:
  flag    all flagged samples count as detected -> flagged share, P, R, F1, FNR, FPR
  ranked  CADE protocol: flagged samples ranked by RAW distance, analyst inspects the top k (k = best-F1 cut-off)
          -> P, R, F1, FNR, FPR at that k, and Effort = k / number of true drifting samples (CADE's normalized effort)

DEFAULT PROTOCOL (used when no --score/--thresholds/--relative is given):
  classical models + PCA baselines   MAD score, T in {0, 2, 2.5, 3, 3.5}                 mode: ranked
  quantum models                     (a) RAW distance, observed range / 7:  T = 0 and T_k = min + k (max - min) / 7
                                     (b) MAD score,    observed range / 7:  same on the MAD score
                                     (c) MAD score, CADE-style fixed T in {0, 1, 1.5, 2, 2.5, 3, 3.5}
                                                                                          mode: both (flag + ranked)
  For quantum models, the threshold SELECTED ON VALIDATION for each scheme (select_threshold.py) is marked with '*'
  as the main result; all thresholds are still reported.

CUSTOM RUNS (any model, any score, any thresholds, any mode):
  --models quantum cade_tuned      only models whose folder name contains one of these
  --score raw|mad                  which score to threshold (with --thresholds or --relative)
  --thresholds 0 1 1.5 2           fixed thresholds
  --relative N                     observed range / N (k = 0 .. N-1, k = 0 means T = 0)
  --mode ranked|flag|both          which result columns
  --by-family                      per-family rows
  --tag NAME                       output file suffix

Usage (from capacity_matched/):
  python3 evaluate.py pca8                                        # full default protocol
  python3 evaluate.py pca64 --by-family
  python3 evaluate.py pca64 --models quantum --score mad --relative 15 --mode both --tag q_mad15
  python3 evaluate.py pca8 --models cade --score mad --thresholds 0 1 2 3 3.5 4 --mode flag --tag cade_flag
Outputs: <experiment>/evaluation/<tag>.txt and .csv  (default tag: protocol)
"""
import os, glob, json, csv, argparse
import numpy as np

CLASSICAL_T = [0, 2, 2.5, 3, 3.5]
QUANTUM_MAD_T = [0, 1, 1.5, 2, 2.5, 3, 3.5]
FLAG_COLS = ['flagged', 'P', 'R', 'F1', 'FNR', 'FPR']
RANK_COLS = ['CADE-P', 'CADE-R', 'CADE-F1', 'CADE-FNR', 'CADE-FPR', 'Effort']


def metrics(score, raw, ood, T):
    flag = (score > T) if T > 0 else np.ones(len(score), bool)
    act = ood.astype(bool); n_ood, n_known = int(act.sum()), int((~act).sum())
    tp, fp, fn = int((flag & act).sum()), int((flag & ~act).sum()), int((~flag & act).sum())
    P = tp / (tp + fp) if tp + fp else 0.0; R = tp / (tp + fn) if tp + fn else 0.0
    out = {'flagged': float(flag.mean()), 'P': P, 'R': R, 'F1': 2 * P * R / (P + R) if P + R else 0.0,
           'FNR': 1.0 - R, 'FPR': float(flag[~act].mean())}
    idx = np.where(flag)[0]; cP = cR = cF = cFPR = eff = 0.0
    if len(idx):
        o = idx[np.argsort(-raw[idx], kind='stable')]; tpc = np.cumsum(ood[o]); k = np.arange(1, len(o) + 1)
        p, r = tpc / k, tpc / n_ood; f1 = np.where(p + r > 0, 2 * p * r / (p + r + 1e-12), 0.0); b = int(np.argmax(f1))
        cP, cR, cF, eff = float(p[b]), float(r[b]), float(f1[b]), float(k[b] / n_ood); cFPR = float((k[b] - tpc[b]) / n_known)
    out.update({'CADE-P': cP, 'CADE-R': cR, 'CADE-F1': cF, 'CADE-FNR': 1.0 - cR, 'CADE-FPR': cFPR, 'Effort': eff})
    return out


def discover(exp):
    models = {}
    for d in sorted(glob.glob(os.path.join(exp, '*', 'results*'))):
        files = sorted(glob.glob(os.path.join(d, 'run*', 'split*', 'ood_scores.csv')))
        if not files:
            continue
        folder, res = os.path.relpath(d, exp).split(os.sep)
        suffix = res[len('results'):].lstrip('_')
        label = folder + ('' if not suffix else (' (L2)' if suffix == 'l2' else f' ({suffix})'))
        tasks = []
        for f in files:
            a = np.genfromtxt(f, delimiter=',', names=True)
            sj = os.path.join(os.path.dirname(f), 'summary.json')
            fam = json.load(open(sj)).get('ood_family', '?') if os.path.exists(sj) else '?'
            tasks.append((fam, a['raw_score'], a['mad_score'], a['is_ood'].astype(int)))
        models[label] = {'folder': folder, 'quantum': 'quantum' in folder.lower(), 'tasks': tasks}
    return models


def thresholds_for(score_vals, scheme):
    kind, arg = scheme
    if kind == 'fixed':
        return [(f'{T:g}', float(T)) for T in arg]
    lo, hi = float(score_vals.min()), float(score_vals.max())
    return [(f'k={k}', 0.0 if k == 0 else lo + k * (hi - lo) / arg) for k in range(arg)]


def run_scheme(model, score, scheme, by_family):
    per = {}
    for fam, raw, mad, ood in model['tasks']:
        sv = raw if score == 'raw' else mad
        for lab, T in thresholds_for(sv, scheme):
            r = metrics(sv, raw, ood, T); r['T'] = T; per.setdefault(lab, []).append((fam, r))
    rows = []
    for lab, lst in per.items():
        cols = list(lst[0][1].keys())
        agg = {c: (np.mean([r[c] for _, r in lst]), np.std([r[c] for _, r in lst])) for c in cols}
        fam_rows = {}
        if by_family:
            for fam in sorted({f for f, _ in lst}):
                sel = [r for f, r in lst if f == fam]
                fam_rows[fam] = {c: (np.mean([r[c] for r in sel]), np.std([r[c] for r in sel])) for c in cols}
        rows.append((lab, agg, fam_rows))
    return rows


def selected(exp, folder):
    p = os.path.join(exp, folder, 'validation', 'selected_thresholds.json')
    return json.load(open(p)) if os.path.exists(p) else {}


def main():
    ap = argparse.ArgumentParser(description='Thresholded detection results (see module docstring).')
    ap.add_argument('experiment')
    ap.add_argument('--models', nargs='+', default=None)
    ap.add_argument('--score', choices=['raw', 'mad'], default=None)
    ap.add_argument('--thresholds', type=float, nargs='+', default=None)
    ap.add_argument('--relative', type=int, default=None)
    ap.add_argument('--mode', choices=['ranked', 'flag', 'both'], default=None)
    ap.add_argument('--by-family', action='store_true')
    ap.add_argument('--tag', default=None)
    a = ap.parse_args()
    exp = os.path.abspath(a.experiment); models = discover(exp)
    if a.models:
        models = {m: v for m, v in models.items() if any(x in m for x in a.models)}
    if not models:
        raise SystemExit(f'No results with ood_scores.csv found under {exp}')
    custom = a.thresholds is not None or a.relative is not None
    if custom and a.score is None:
        raise SystemExit('--thresholds / --relative need --score raw|mad')
    tag = a.tag or ('custom' if custom else 'protocol')
    lines, rows_csv = [], []
    def log(s=''):
        print(s); lines.append(s)
    log('=' * 124)
    log(f'DETECTION RESULTS -- {os.path.basename(exp)}  ({"custom run" if custom else "default protocol"}; test scores; mean +- std over tasks)')
    log('=' * 124)
    for m, model in models.items():
        if custom:
            schemes = [(a.score, ('fixed', a.thresholds) if a.thresholds is not None else ('rel', a.relative), None)]
            mode = a.mode or 'both'
        elif model['quantum']:
            schemes = [('raw', ('rel', 7), 'raw_rel7'), ('mad', ('rel', 7), 'mad_rel7'), ('mad', ('fixed', QUANTUM_MAD_T), 'mad_fixed')]
            mode = a.mode or 'both'
        else:
            schemes = [('mad', ('fixed', CLASSICAL_T), None)]
            mode = a.mode or 'ranked'
        cols = (FLAG_COLS if mode in ('flag', 'both') else []) + (RANK_COLS if mode in ('ranked', 'both') else [])
        sel = selected(exp, model['folder']) if model['quantum'] else {}
        log(f'\n##### {m}  ({len(model["tasks"])} tasks){"  [quantum]" if model["quantum"] else "  [classical / baseline: CADE protocol]"}')
        for score, scheme, key in schemes:
            desc = f'{"RAW distance" if score == "raw" else "MAD score"}, ' + \
                   ('fixed T' if scheme[0] == 'fixed' else f'observed range / {scheme[1]} (k = 0 is T = 0, flag all)')
            main_lab = None
            if key and key in sel:
                main_lab = f'k={sel[key]}' if scheme[0] == 'rel' else f'{float(sel[key]):g}'
            log(f'\n  {desc}' + (f'   [MAIN = {main_lab}, selected on validation]' if main_lab else ''))
            log(f"    {'threshold':<11}{'T':>9}" + ''.join(f'{c:>16}' for c in cols))
            sch = scheme[0] + (str(scheme[1]) if scheme[0] == 'rel' else '')
            for lab, agg, fam_rows in run_scheme(model, score, scheme, a.by_family):
                star = ' *' if lab == main_lab else ''
                log(f'    {lab + star:<11}{agg["T"][0]:>9.3f}' + ''.join(f'{agg[c][0]:>9.3f}+-{agg[c][1]:<5.3f}' for c in cols))
                rows_csv.append({'model': m, 'score': score, 'scheme': sch, 'threshold': lab, 'T': agg['T'][0], 'family': 'ALL',
                                 'main': int(bool(star)), **{c: agg[c][0] for c in cols}, **{c + '_std': agg[c][1] for c in cols}})
                for fam, fr in fam_rows.items():
                    log(f'      {fam:<16}' + ''.join(f'{fr[c][0]:>9.3f}+-{fr[c][1]:<5.3f}' for c in cols))
                    rows_csv.append({'model': m, 'score': score, 'scheme': sch, 'threshold': lab, 'T': fr['T'][0], 'family': fam,
                                     'main': int(bool(star)), **{c: fr[c][0] for c in cols}, **{c + '_std': fr[c][1] for c in cols}})
    log('\n  flag: all samples with score > T count as detected (T = 0 flags every sample): flagged share, P, R, F1, FNR, FPR.')
    log('  ranked (CADE protocol): flagged samples ranked by raw distance, top k inspected (k = best-F1 cut-off):')
    log('  CADE-P/R/F1/FNR/FPR at that k; Effort = k / number of true drifting samples.  * = threshold selected on validation.')
    out = os.path.join(exp, 'evaluation'); os.makedirs(out, exist_ok=True)
    open(os.path.join(out, f'{tag}.txt'), 'w').write('\n'.join(lines) + '\n')
    lead = ['model', 'score', 'scheme', 'threshold', 'T', 'family', 'main']
    keys = lead + sorted({k for r in rows_csv for k in r} - set(lead))
    with open(os.path.join(out, f'{tag}.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys, restval=''); w.writeheader(); w.writerows(rows_csv)
    print(f'\nSaved: {out}/{tag}.txt and .csv')


if __name__ == '__main__':
    main()
