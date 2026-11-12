#!/usr/bin/env python3
"""
fpr_ranked_f1_pca8.py -- Experiments2/EMBER/pca8: EMBER 2018 (8 families, capped 1000, CADE per-split cut)   (standalone analysis, no retraining; needs threshold_sweep_pca8.py next to it)
====================================================================================================================
CADE's ranked F1 under a false-positive budget, on the raw drift score.

CADE ranks test samples by distance and reports the best F1 of an analyst who inspects the list from the top.
Here the analyst may stop only at cut-offs whose false positive rate on the known families does not exceed a budget
f in {10, 20, 30, 40, 100}% (100% = no budget = CADE's ranked F1 on the full list = TF-F1).
  test-calibrated  : the budget is enforced exactly on the known test samples (cut-offs with FP / n_known <= f)
  train-calibrated : the list is cut at the (100(1-f))-th percentile of the model's TRAINING raw scores
                     (deployable; identical to the raw threshold sweep), reported for comparison
At each budget: ranked F1, precision, recall, FPR at the best stopping point, and CADE's normalized inspection
effort (# inspected / # drifting samples).

Outputs: <model>/results/FPR_ranked_F1/{per_task.csv, aggregate.csv, summary.txt}
         pca8/fpr_ranked_f1_summary/{all_models.csv, summary.txt, table_fpr_ranked_f1_pca8_p702.tex, plots/*.png}
PCA-only folders are skipped.   Usage (from EMBER/pca8/):  python3 fpr_ranked_f1_pca8.py [--models ...]
"""
import os, sys, glob, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import threshold_sweep_pca8 as ts

BUDGETS = [0.10, 0.20, 0.30, 0.40, 1.00]
STATS = ['F1', 'Precision', 'Recall', 'FPR', 'Effort']
P702 = [('quantum_pca8/pennylane/quantum_hybrid_v7_p702', 'Quantum hybrid', 702),
        ('quantum_pca8/pennylane/quantum_amp_fixed_p702', 'Quantum amplitude (fixed)', 702),
        ('twin_pca8_p702/classical_fidtwin_c_702', 'Fidelity twin, complex', 709),
        ('twin_pca8_p702/classical_fidtwin_p_702', 'Fidelity twin, plain', 709),
        ('twin_pca8_p702/classical_fidtwin_r_702', 'Fidelity twin, real', 708),
        ('cade_pca8_p702/cade_full_8_702', 'CADE-mini, full', 719),
        ('cade_pca8_p702/cade_enc_8_702', 'CADE-mini, encoder', 711),
        ('cade_pca8_p702/cade_tuned_8_702', 'CADE-mini, tuned', 711),
        ('cade_pca8_p702/cade_deep_8_702', 'CADE-mini, deep', 710)]
bl = lambda f: f'{100 * f:g}'


def ranked_under_budget(s, ood, allowed):
    """Best F1 over cut-offs k (top-k of the ranking by s) for which allowed[k-1] is True."""
    order = np.argsort(-s, kind='stable'); o = ood[order].astype(int)
    tp = np.cumsum(o); k = np.arange(1, len(s) + 1); fp = k - tp
    n_ood, n_known = int(ood.sum()), int((~ood).sum())
    p, r = tp / k, tp / n_ood; f = np.where(p + r > 0, 2 * p * r / (p + r + 1e-12), 0.0)
    f = np.where(allowed(fp, n_known, s[order]), f, -1.0)
    if f.max() < 0:
        return dict(F1=0.0, Precision=0.0, Recall=0.0, FPR=0.0, Effort=0.0)
    b = int(np.argmax(f))
    return dict(F1=float(f[b]), Precision=float(p[b]), Recall=float(r[b]), FPR=float(fp[b] / n_known),
                Effort=float(k[b] / n_ood))


def task_metrics(t):
    s, ood = t['raw_te'], t['is_ood'].astype(bool); row = {}
    for f in BUDGETS:
        te = ranked_under_budget(s, ood, lambda fp, nk, sv, f=f: fp <= f * nk + 1e-9)
        if f >= 1.0:
            tr = te
        else:
            tau = np.percentile(t['raw_tr'], 100 * (1 - f))
            tr = ranked_under_budget(s, ood, lambda fp, nk, sv, tau=tau: sv > tau)
        row.update({f'test_{k}@{bl(f)}': v for k, v in te.items()}); row.update({f'train_{k}@{bl(f)}': v for k, v in tr.items()})
        known = np.sort(s[~ood])[::-1]; kk = int(np.floor(f * len(known)))
        thr = known[kk] if kk < len(known) else -np.inf
        row[f'test_TPR@{bl(f)}'] = float((s[ood] > thr).mean())          # recall when exactly <= f of known are flagged
    return row


def run_model(model_dir):
    rel = os.path.relpath(model_dir, HERE); rows = []
    for td in sorted(glob.glob(os.path.join(model_dir, 'results', 'run*', 'split*'))):
        t, _ = ts.load_task(td)
        if t is None:
            continue
        r_, s_ = td.rstrip('/').split(os.sep)[-2:]
        try:
            fam = json.load(open(os.path.join(td, 'summary.json'))).get('ood_family', '')
        except Exception:
            fam = ''
        rows.append({'run': r_, 'split': s_, 'ood_family': fam, **task_metrics(t)})
    if not rows:
        print(f'== {rel}: no usable tasks'); return None
    d = os.path.join(model_dir, 'results', 'FPR_ranked_F1'); os.makedirs(d, exist_ok=True)
    ts._write_csv(os.path.join(d, 'per_task.csv'), rows)
    agg = {'model': rel, 'n_tasks': len(rows)}
    for k in [k for k in rows[0] if k not in ('run', 'split', 'ood_family')]:
        v = np.array([r[k] for r in rows], float); agg[f'{k}_mean'], agg[f'{k}_std'] = float(v.mean()), float(v.std())
    ts._write_csv(os.path.join(d, 'aggregate.csv'), [agg])
    pm = lambda k: f'{agg[k + "_mean"]:.3f}+-{agg[k + "_std"]:.3f}'
    L = [f'{rel}  ({len(rows)} tasks)  ranked F1 under an FPR budget, mean +- std', '',
         f'{"budget":>7} | {"F1 (test-cal.)":>14} {"Prec":>13} {"Recall":>13} {"FPR@stop":>13} {"Effort":>13} | {"F1 (train-cal.)":>15} {"FPR@stop":>13}']
    for f in BUDGETS:
        b = bl(f)
        L.append(f'{(b + "%") if f < 1 else "none":>7} | {pm("test_F1@" + b):>14} {pm("test_Precision@" + b):>13} '
                 f'{pm("test_Recall@" + b):>13} {pm("test_FPR@" + b):>13} {pm("test_Effort@" + b):>13} | '
                 f'{pm("train_F1@" + b):>15} {pm("train_FPR@" + b):>13}')
    open(os.path.join(d, 'summary.txt'), 'w').write('\n'.join(L) + '\n'); print('\n' + '\n'.join(L))
    return agg


def write_summary(aggs):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    d = os.path.join(HERE, 'fpr_ranked_f1_summary'); os.makedirs(os.path.join(d, 'plots'), exist_ok=True)
    ts._write_csv(os.path.join(d, 'all_models.csv'), aggs)
    bs = [bl(f) for f in BUDGETS]; hdr = ' '.join(f'{(b + "%") if b != "100" else "none":>22}' for b in bs)
    L = []
    for cal, name in [('test', 'test-calibrated (FPR budget enforced exactly on the known test samples)'),
                      ('train', 'train-calibrated (list cut at the training-score percentile)')]:
        L += [f'Ranked F1 under an FPR budget, {name}: F1 [FPR at the best stop]' + (' (TPR at budget)' if cal == 'test' else ''), f'{"model":<50} {hdr}']
        for a in aggs:
            L.append(f'{a["model"]:<50} ' + ' '.join((f'{a[f"{cal}_F1@{b}_mean"]:.3f} [{a[f"{cal}_FPR@{b}_mean"]:.2f}]' +
                     (f' ({a[f"test_TPR@{b}_mean"]:.2f})' if cal == 'test' and b != '100' else '')).rjust(22) for b in bs))
        L.append('')
    open(os.path.join(d, 'summary.txt'), 'w').write('\n'.join(L) + '\n'); print('\n' + '\n'.join(L))
    for cal in ['test', 'train']:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for a in aggs:
            ax.plot([100 * f for f in BUDGETS], [a[f'{cal}_F1@{b}_mean'] for b in bs], '-o', ms=3, label=a['model'].split('/')[-1])
        ax.set_xscale('log'); ax.set_xticks([100 * f for f in BUDGETS]); ax.set_xticklabels([f'{b}%' if b != '100' else 'none' for b in bs])
        ax.set_xlabel(f'FPR budget ({cal}-calibrated)'); ax.set_ylabel('ranked F1'); ax.set_ylim(0, 1); ax.grid(alpha=0.3)
        ax.legend(fontsize=6, ncol=2); fig.tight_layout(); fig.savefig(os.path.join(d, 'plots', f'ranked_f1_vs_budget_{cal}.png'), dpi=150)
        plt.close(fig)
    byf = {a['model']: a for a in aggs}; rows = [(lab, p, byf[f]) for f, lab, p in P702 if f in byf]
    if rows:
        cols = ['10', '20', '30', '40', '100']
        best = {c: max(a[f'test_F1@{c}_mean'] for _, _, a in rows) for c in cols}
        cell = lambda a, c: (r'\textbf{%.3f}' if abs(a[f'test_F1@{c}_mean'] - best[c]) < 5e-4 else '%.3f') % a[f'test_F1@{c}_mean'] + \
            r'\,{\scriptsize$\pm$%.3f}' % a[f'test_F1@{c}_std'] + (r' {\scriptsize(%.2f)}' % a[f'test_TPR@{c}_mean'] if c != '100' else r' {\scriptsize[%.2f]}' % a[f'test_FPR@{c}_mean'])
        T = [r'% Requires \usepackage{booktabs} and \usepackage{graphicx}', r'\begin{table*}[t]', r'  \centering',
             r'  \caption{Ranked F1 under a false positive budget for the PCA-8 models with about 702 parameters. Following',
             r'  CADE, test samples are ranked by their raw drift score and an analyst inspects the list from the top; the',
             r'  reported F1 is the best F1 among stopping points whose false positive rate on the known families does not',
             "  exceed the budget. ``None'' imposes no budget and equals CADE's ranked F1 on the full list. Mean $\\pm$",
             r'  standard deviation over 24 tasks. Parentheses: recall (TPR) at the budget; brackets: FPR at the best stopping point.}',
             r'  \label{tab:fpr_ranked_f1_pca8_p702}', r'  \small', r'  \resizebox{\textwidth}{!}{%',
             r'  \begin{tabular}{@{}lrccccc@{}}', r'    \toprule',
             r'    & & \multicolumn{5}{c}{Ranked F1 at FPR budget} \\', r'    \cmidrule(l){3-7}',
             r'    Model & Params & 10\% & 20\% & 30\% & 40\% & None \\', r'    \midrule']
        prev = None
        for lab, p, a in rows:
            g = lab.split(',')[0]
            if prev and g != prev:
                T.append(r'    \midrule')
            T.append(f'    {lab} & {p} & ' + ' & '.join(cell(a, c) for c in cols) + r' \\'); prev = g
        T += [r'    \bottomrule', r'  \end{tabular}}', r'\end{table*}']
        open(os.path.join(d, 'table_fpr_ranked_f1_pca8_p702.tex'), 'w').write('\n'.join(T) + '\n'); print('\n'.join(T))
    print(f'\nSaved: {os.path.relpath(d, HERE)}/{{all_models.csv, summary.txt, table_fpr_ranked_f1_pca8_p702.tex, plots/}}')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--models', nargs='*'); ap.add_argument('--include-ablation', action='store_true')
    a = ap.parse_args()
    models = [os.path.join(HERE, m) for m in a.models] if a.models else \
        ts.find_models(ts.DEFAULT_GROUPS + (['ablation_pca8'] if a.include_ablation else []))
    aggs = [x for x in (run_model(m) for m in models if 'pca_only' not in m) if x]
    if aggs:
        write_summary(aggs)


if __name__ == '__main__':
    main()
