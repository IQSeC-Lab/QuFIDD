#!/usr/bin/env python3
"""
cade_protocol_eval.py
=====================
Re-scores a finished experiment group with CADE's own drift-detection
evaluation protocol (Yang et al., USENIX Security 2021), using the per-sample
scores already saved in results/run*/split*/ood_scores.csv. No retraining.

CADE protocol (Algorithm 1, Sec. 3.2 "Ranking Drifting Samples", Sec. 4.2):
  1. Flag a test sample as a candidate drifting sample if its MAD anomaly
     index A = min_c |d_c - median_c| / MAD_c exceeds T_MAD.
  2. Rank the flagged samples by raw distance to the nearest centroid,
     farthest first.
  3. Walk down the ranked list (simulated analyst) and report the HIGHEST F1,
     with its precision, recall, FPR, FNR and normalized inspection effort
     (inspected samples / true OOD samples).
T_MAD is swept over {0, 2.0, 2.5, 3.0, 3.5} as in CADE's Table 9.
T_MAD = 0 means no MAD filtering (CADE's setting for its Vanilla-AE baseline,
Appendix B). T_MAD = 3.5 is CADE's default.

Also reports threshold-free ROC-AUC / PR-AUC of the raw score and of the MAD
score over ALL test samples, for reference.

Layout (one copy of this script per group):
  GroupXX/
    results/run{0,1,2}/split{0..}/ood_scores.csv   <- read
    cade_protocol_eval/
      cade_protocol_eval.py                        <- this script
      run0/split0/cade_protocol.json               <- written (per task)
      run0_results.csv, run1_results.csv, ...      <- written (per run)
      aggregate.csv                                <- written (all runs)
      aggregate_summary.txt                        <- written (printed report)

Usage (from inside GroupXX/cade_protocol_eval/):
  python3 cade_protocol_eval.py
  python3 cade_protocol_eval.py --data-dir ~/projects/OOD/Data/cade_lamda_4class_vt
"""

import os, sys, csv, json, argparse
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

THRESHOLDS = [0.0, 2.0, 2.5, 3.0, 3.5]
METRICS    = ['F1', 'Precision', 'Recall', 'FPR', 'FNR', 'Effort', 'Frac_flagged']
SEEDS      = {0: 42, 1: 123, 2: 456}

HERE           = os.path.dirname(os.path.abspath(__file__))
GROUP_DIR      = os.path.dirname(HERE)
GROUP_NAME     = os.path.basename(GROUP_DIR)
DEFAULT_RESULT = os.path.join(GROUP_DIR, 'results')
DEFAULT_DATA   = os.path.expanduser('~/projects/OOD/Data/cade_lamda_4class_vt')


class Tee:
    """Print to stdout and collect lines for aggregate_summary.txt."""
    def __init__(self):
        self.lines = []
    def __call__(self, s=''):
        print(s, flush=True)
        self.lines.append(s)


def cade_eval(raw, mad, is_ood, T):
    """CADE protocol at one T_MAD. Returns a dict of metrics."""
    n_ood   = int(is_ood.sum())
    n_known = int(len(is_ood) - n_ood)
    flagged = np.ones(len(raw), bool) if T == 0 else (mad > T)
    idx     = np.where(flagged)[0]
    out = {'T_MAD': T, 'n_flagged': int(len(idx)),
           'Frac_flagged': float(len(idx) / len(raw))}
    if len(idx) == 0 or n_ood == 0:
        out.update(F1=0.0, Precision=0.0, Recall=0.0, FPR=0.0, FNR=1.0,
                   Effort=0.0, k_best=0, TP=0, FP=0)
        return out
    order = idx[np.argsort(-raw[idx], kind='stable')]   # farthest first
    tp    = np.cumsum(is_ood[order])
    k     = np.arange(1, len(order) + 1)
    prec  = tp / k
    rec   = tp / n_ood
    f1    = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec + 1e-12), 0.0)
    b     = int(np.argmax(f1))
    TP, FP = int(tp[b]), int(k[b] - tp[b])
    out.update(F1=float(f1[b]), Precision=float(prec[b]), Recall=float(rec[b]),
               FPR=float(FP / n_known) if n_known else 0.0,
               FNR=float(1 - rec[b]), Effort=float(k[b] / n_ood),
               k_best=int(k[b]), TP=TP, FP=FP)
    return out


def load_scores(path):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        raise ValueError(f'empty file: {path}')
    raw_key = 'raw_score' if 'raw_score' in rows[0] else 'raw_euclidean_score'
    raw = np.array([float(r[raw_key]) for r in rows])
    mad = np.array([float(r['mad_score']) for r in rows])
    return raw, mad


def summary_tf_f1(task_dir):
    """Original threshold-free F1 from summary.json, for the T=0 sanity check."""
    p = os.path.join(task_dir, 'summary.json')
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    for sec, key in [('Threshold_Free', 'best_f1'), ('CADE_Threshold_Free', 'F1')]:
        if sec in d and key in d[sec]:
            return float(d[sec][key])
    return None


def main():
    ap = argparse.ArgumentParser(description='CADE-protocol re-scoring of saved OOD scores')
    ap.add_argument('--results-dir', default=DEFAULT_RESULT)
    ap.add_argument('--data-dir',    default=DEFAULT_DATA)
    ap.add_argument('--prefix',      default='drebin_new')
    ap.add_argument('--out-dir',     default=HERE)
    args = ap.parse_args()
    res_dir  = os.path.expanduser(args.results_dir)
    data_dir = os.path.expanduser(args.data_dir)
    out_dir  = os.path.expanduser(args.out_dir)

    log = Tee()
    log('=' * 90)
    log(f'{GROUP_NAME} -- CADE-protocol evaluation (flag A > T_MAD, rank flagged by raw '
        f'distance, best F1 down the list)')
    log(f'results: {res_dir}')
    log(f'data:    {data_dir}')
    log('=' * 90)

    # discover splits from run0
    n_splits = 0
    while os.path.isdir(os.path.join(res_dir, 'run0', f'split{n_splits}')):
        n_splits += 1
    if n_splits == 0:
        sys.exit(f'No results found under {res_dir}')

    all_rows, missing, sanity = [], [], []
    for r in range(3):
        run_rows = []
        for s in range(n_splits):
            task_dir = os.path.join(res_dir, f'run{r}', f'split{s}')
            sc_path  = os.path.join(task_dir, 'ood_scores.csv')
            if not os.path.exists(sc_path):
                missing.append((r, s))
                continue
            z   = np.load(os.path.join(data_dir, f'{args.prefix}_{s}.npz'), allow_pickle=True)
            ytr, yte = z['y_train'], z['y_test']
            fam = str(z['heldout_family']) if 'heldout_family' in z.files else f'split{s}'
            is_ood = (~np.isin(yte, np.unique(ytr))).astype(int)

            raw, mad = load_scores(sc_path)
            if len(raw) != len(yte):
                sys.exit(f'Length mismatch in {sc_path}: {len(raw)} scores vs {len(yte)} labels')
            n_nan = int(np.isnan(mad).sum())
            if n_nan:
                log(f'  WARNING run{r}/split{s}: {n_nan}/{len(mad)} NaN MAD scores '
                    f'(treated as 0 = never flagged)')
            mad = np.nan_to_num(mad, nan=0.0, posinf=1e12)

            ref = {'ROC_AUC_raw': float(roc_auc_score(is_ood, raw)),
                   'PR_AUC_raw':  float(average_precision_score(is_ood, raw)),
                   'ROC_AUC_mad': float(roc_auc_score(is_ood, mad)),
                   'PR_AUC_mad':  float(average_precision_score(is_ood, mad))}
            per_T = [cade_eval(raw, mad, is_ood, T) for T in THRESHOLDS]

            # sanity: T=0 should reproduce the script's own threshold-free F1
            orig = summary_tf_f1(task_dir)
            if orig is not None:
                sanity.append((r, s, per_T[0]['F1'], orig))

            task = {'group': GROUP_NAME, 'run_id': r, 'split_id': s,
                    'seed': SEEDS.get(r), 'ood_family': fam,
                    'n_ood': int(is_ood.sum()), 'n_known': int(len(is_ood) - is_ood.sum()),
                    'n_nan_mad': n_nan, 'reference_threshold_free': ref,
                    'cade_protocol': per_T}
            tdir = os.path.join(out_dir, f'run{r}', f'split{s}')
            os.makedirs(tdir, exist_ok=True)
            with open(os.path.join(tdir, 'cade_protocol.json'), 'w') as f:
                json.dump(task, f, indent=2)

            for e in per_T:
                row = {'run_id': r, 'split_id': s, 'seed': SEEDS.get(r), 'ood_family': fam,
                       'T_MAD': e['T_MAD'], 'n_flagged': e['n_flagged'],
                       'k_best': e['k_best'], 'TP': e['TP'], 'FP': e['FP']}
                row.update({m: round(e[m], 6) for m in METRICS})
                row.update({k: round(v, 6) for k, v in ref.items()})
                run_rows.append(row)
        if run_rows:
            with open(os.path.join(out_dir, f'run{r}_results.csv'), 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(run_rows[0].keys()))
                w.writeheader(); w.writerows(run_rows)
        all_rows += run_rows

    if not all_rows:
        sys.exit('No tasks evaluated.')
    n_tasks = len({(x['run_id'], x['split_id']) for x in all_rows})
    log(f'Tasks evaluated: {n_tasks}/{3 * n_splits}   Missing: {missing if missing else "none"}')

    if sanity:
        worst = max(abs(a - b) for _, _, a, b in sanity)
        log(f'Sanity check (T_MAD=0 F1 vs threshold-free F1 in summary.json): '
            f'max |diff| = {worst:.4f} over {len(sanity)} tasks'
            + ('  OK' if worst < 0.002 else '  <-- CHECK: ranking may differ from the original script'))

    # ---- aggregate ------------------------------------------------------
    fams = []
    for x in all_rows:
        if x['ood_family'] not in fams:
            fams.append(x['ood_family'])
    agg_rows = []

    def stats(sel, key):
        v = np.array([x[key] for x in sel], float)
        return float(v.mean()), float(v.std())

    log('')
    log('Overall (mean +- std over all tasks; std = population std)')
    log(f"{'T_MAD':<7}" + ''.join(f'{m:>17}' for m in METRICS))
    for T in THRESHOLDS:
        sel = [x for x in all_rows if x['T_MAD'] == T]
        line = f'{T:<7}'
        rec = {'scope': 'ALL', 'ood_family': 'ALL', 'T_MAD': T, 'n_tasks': len(sel)}
        for m in METRICS:
            mu, sd = stats(sel, m)
            line += f'{mu:>9.3f}+-{sd:<6.3f}'
            rec[f'{m}_mean'], rec[f'{m}_std'] = round(mu, 6), round(sd, 6)
        log(line)
        agg_rows.append(rec)

    log('')
    log('Per family: F1 (mean +- std over runs) at each T_MAD')
    log(f"{'family':<16}{'n':>3}" + ''.join(f'{"T=" + str(T):>16}' for T in THRESHOLDS))
    for fam in fams:
        line, n = f'{fam:<16}', 0
        for T in THRESHOLDS:
            sel = [x for x in all_rows if x['ood_family'] == fam and x['T_MAD'] == T]
            n = len(sel)
            rec = {'scope': 'family', 'ood_family': fam, 'T_MAD': T, 'n_tasks': n}
            for m in METRICS:
                mu, sd = stats(sel, m)
                rec[f'{m}_mean'], rec[f'{m}_std'] = round(mu, 6), round(sd, 6)
            agg_rows.append(rec)
            line += f"{rec['F1_mean']:>9.3f}+-{rec['F1_std']:<5.3f}"
        log(line[:16] + f'{n:>3}' + line[16:])

    log('')
    log('Reference: threshold-free AUCs over all test samples (mean +- std over runs)')
    log(f"{'family':<16}{'ROC_raw':>15}{'ROC_mad':>15}{'PR_raw':>15}{'PR_mad':>15}")
    ref_rows = [x for x in all_rows if x['T_MAD'] == 0.0]
    for fam in fams + ['ALL']:
        sel = ref_rows if fam == 'ALL' else [x for x in ref_rows if x['ood_family'] == fam]
        vals = [stats(sel, k) for k in ['ROC_AUC_raw', 'ROC_AUC_mad', 'PR_AUC_raw', 'PR_AUC_mad']]
        log(f'{fam:<16}' + ''.join(f'{mu:>8.3f}+-{sd:<5.3f}' for mu, sd in vals))

    with open(os.path.join(out_dir, 'aggregate.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
        w.writeheader(); w.writerows(agg_rows)
    with open(os.path.join(out_dir, 'aggregate_summary.txt'), 'w') as f:
        f.write('\n'.join(log.lines) + '\n')
    print(f'\nSaved per-task JSON under {out_dir}/run*/split*/, run*_results.csv, '
          f'aggregate.csv, aggregate_summary.txt')


if __name__ == '__main__':
    main()
