#!/usr/bin/env python3
"""
ensemble_cross.py -- capacity_matched / capacity_matched_8fam (top level, next to evaluate.py)
======================================================================================================================
PROOF OF CONCEPT, from saved scores only (nothing is retrained): the PCA-64 AMPLITUDE view (amp-64, the best
directional detector) combined with the PCA-8 ANGLE view (angle-8, the component-value view), same seed and split.

RULE (as ensemble_angle_amp.py, fixed in advance): each model's raw distance is standardized by its own TRAINING raw
scores, z = (d - median_train) / (1.4826 MAD_train) (signed); the ensemble score is the MEAN of the two z-scores.

COMPARED WITH (same rule, same tasks):
  members alone           amp-64, angle-8
  Q8 reference            angle-8 + amp-8 (the PCA-8 ensemble)
  CLASSICAL cross pairs   CADE-mini PCA-64 + CADE-mini PCA-8 of the same variant (enc + enc, tuned + tuned,
                          full + full): the like-for-like classical counterpart (one PCA-64 and one PCA-8 model)
  classical singles       CADE-mini PCA-64 encoder-only and tuned
Parameter counts are read from each model's saved summary.json.
STATISTICS: per-task Wilcoxon + hierarchical bootstrap (families -> seeds -> test samples) of the ROC-AUC difference.
WEIGHT CURVE: w * z_first + (1 - w) * z_second for w = 0, 0.1, ..., 1 -- DESCRIPTIVE ONLY, no weight is selected.

PROOF-OF-CONCEPT CRITERION (fixed in advance): amp-64 + angle-8 beats amp-64 alone AND the best classical cross pair
under the hierarchical bootstrap.

Usage (from capacity_matched/ or capacity_matched_8fam/):
  python3 ensemble_cross.py                 # default: pca64 + pca8, n_boot 500
  python3 ensemble_cross.py --n-boot 200
Output: pca64/comparison/ensemble_cross.txt, ensemble_cross_per_task.csv, ensemble_cross_curve.csv,
        plots/ensemble_cross_curve.png
"""
import os, glob, json, csv, argparse
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import wilcoxon


def find(root, prefix):
    m = sorted(f for f in os.listdir(root) if f.startswith(prefix) and os.path.isdir(os.path.join(root, f, 'results')))
    return os.path.join(root, m[0]) if m else None


def load_task(folder, run, split):
    d = os.path.join(folder, 'results', f'run{run}', f'split{split}')
    if not os.path.exists(os.path.join(d, 'ood_scores.csv')):
        return None
    c = np.genfromtxt(os.path.join(d, 'ood_scores.csv'), delimiter=',', names=True)
    s = json.load(open(os.path.join(d, 'summary.json')))
    return {'raw': c['raw_score'], 'ood': c['is_ood'].astype(int), 'label': c['true_label'],
            'train': np.load(os.path.join(d, 'train_raw_scores.npy')), 'family': s.get('ood_family', f'split{split}'),
            'params': s.get('n_trainable_params')}


def zscore(t):
    med = np.median(t['train']); mad = 1.4826 * np.median(np.abs(t['train'] - med))
    return (t['raw'] - med) / (mad if mad > 0 else 1e-12)


def ranked(score, ood):
    o = np.argsort(-score); y = ood[o]; tp = np.cumsum(y); k = np.arange(1, len(y) + 1)
    n_pos, n_neg = int(ood.sum()), int(len(ood) - ood.sum())
    P, R = tp / k, tp / max(n_pos, 1); F = np.where(P + R > 0, 2 * P * R / (P + R + 1e-12), 0)
    i = int(np.argmax(F))
    return {'F1': F[i], 'P': P[i], 'R': R[i], 'FPR': (k[i] - tp[i]) / max(n_neg, 1), 'FNR': 1 - R[i], 'Effort': k[i] / max(n_pos, 1)}


def evaluate(score, ood):
    e = {'ROC': roc_auc_score(ood, score), 'PR': average_precision_score(ood, score)}; e.update(ranked(score, ood)); return e


def hier_boot(A, B, oods, fams, n_boot, seed=0):
    rng = np.random.default_rng(seed); by = {}
    for i, f in enumerate(fams):
        by.setdefault(f, []).append(i)
    keys = list(by); diffs = []
    for _ in range(n_boot):
        vals = []
        for f in rng.choice(keys, len(keys)):
            for i in rng.choice(by[f], len(by[f])):
                n = len(oods[i]); idx = rng.integers(0, n, n); o = oods[i][idx]
                if o.min() != o.max():
                    vals.append(roc_auc_score(o, A[i][idx]) - roc_auc_score(o, B[i][idx]))
        diffs.append(np.mean(vals))
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--amp-dir', default='pca64'); ap.add_argument('--angle-dir', default='pca8')
    ap.add_argument('--n-boot', type=int, default=500); ap.add_argument('--curve-step', type=float, default=0.1)
    a = ap.parse_args(); R64, R8 = os.path.abspath(a.amp_dir), os.path.abspath(a.angle_dir)
    F = {'amp-64': find(R64, 'quantum_amp_'), 'angle-8': find(R8, 'quantum_angle_'), 'amp-8': find(R8, 'quantum_amp_')}
    for v in ('enc', 'tuned', 'full'):
        F[f'{v}-64'], F[f'{v}-8'] = find(R64, f'cade_{v}'), find(R8, f'cade_{v}')
    if not (F['amp-64'] and F['angle-8']):
        raise SystemExit(f'Need quantum_amp_* in {R64} and quantum_angle_* in {R8}')
    ens = {'amp-64 alone': ['amp-64'], 'angle-8 alone': ['angle-8'], 'Q: amp-64 + angle-8': ['amp-64', 'angle-8']}
    if F['amp-8']:
        ens['Q8: amp-8 + angle-8'] = ['amp-8', 'angle-8']
    for v in ('enc', 'tuned'):
        if F[f'{v}-64']:
            ens[f'{v}-64 alone'] = [f'{v}-64']
    for v in ('enc', 'tuned', 'full'):
        if F[f'{v}-64'] and F[f'{v}-8']:
            ens[f'C: {v}-64 + {v}-8'] = [f'{v}-64', f'{v}-8']
    runs = sorted({int(p.split('run')[-1]) for p in glob.glob(os.path.join(F['amp-64'], 'results', 'run*'))})
    splits = sorted({int(p.split('split')[-1]) for p in glob.glob(os.path.join(F['amp-64'], 'results', 'run*', 'split*'))})
    lines, rows = [], []
    def log(s=''):
        print(s, flush=True); lines.append(s)
    log('=' * 118); log(f'CROSS-INPUT ENSEMBLE: amp-64 (amplitude, PCA-64) + angle-8 (angle, PCA-8) -- {os.path.dirname(R64)}'); log('=' * 118)
    Z = {k: [] for k in ens}; oods, fams, tasks, params = [], [], [], {}
    for s in range(len(splits)):
        for r in runs:
            loaded = {m: load_task(F[m], r, splits[s]) for m in {m for v in ens.values() for m in v}}
            base = loaded['amp-64']
            if base is None or any(t is None or not np.array_equal(t['label'], base['label']) for t in loaded.values()):
                log(f'  run{r} split{splits[s]}: a member is missing or its test set differs -- task skipped'); continue
            for m, t in loaded.items():
                params.setdefault(m, t['params'])
            zs = {m: zscore(t) for m, t in loaded.items()}
            for name, mem in ens.items():
                Z[name].append(np.mean([zs[m] for m in mem], axis=0))
            oods.append(base['ood']); fams.append(base['family']); tasks.append((r, splits[s]))
    order = ['amp-64', 'angle-8', 'amp-8', 'enc-64', 'tuned-64', 'full-64', 'enc-8', 'tuned-8', 'full-8']
    log('  members: ' + ', '.join(f'{m} = {os.path.relpath(F[m], os.path.dirname(R64))} ({params.get(m)} params)'
                                  for m in order if m in params))
    pc = lambda mem: sum(params.get(m) or 0 for m in mem)
    log(f'  tasks: {len(tasks)} ({len(set(fams))} families x {len(runs)} seeds); rule: mean of the members\' training-standardized z-scores')
    cols = ['ROC', 'PR', 'F1', 'P', 'R', 'FPR', 'FNR', 'Effort']
    res = {k: [evaluate(z, o) for z, o in zip(v, oods)] for k, v in Z.items()}
    log(f'\nOVERALL (mean over {len(tasks)} tasks)')
    log(f"  {'model / ensemble':<26}{'params':>8}" + ''.join(f'{c:>9}' for c in cols))
    for name, mem in ens.items():
        log(f'  {name:<26}{pc(mem):>8}' + ''.join(f'{np.mean([e[c] for e in res[name]]):>9.3f}' for c in cols))
    fl = sorted(set(fams)); names = list(ens)
    log('\nPER FAMILY -- raw ROC-AUC (mean over seeds)')
    log(f"  {'family':<14}" + ''.join(f'{n[:17]:>18}' for n in names))
    for f in fl:
        idx = [i for i, x in enumerate(fams) if x == f]
        log(f'  {f:<14}' + ''.join(f'{np.mean([res[n][i]["ROC"] for i in idx]):>18.3f}' for n in names))
    Q = 'Q: amp-64 + angle-8'
    key = [(Q, 'amp-64 alone'), (Q, 'angle-8 alone')] + [(Q, n) for n in names if n.startswith('C:') or n.endswith('-64 alone') and n != 'amp-64 alone']
    if 'Q8: amp-8 + angle-8' in ens:
        key.append((Q, 'Q8: amp-8 + angle-8'))
    log(f'\nKEY COMPARISONS -- raw ROC-AUC (A minus B); hierarchical bootstrap n = {a.n_boot}')
    log(f"  {'A':<22}{'B':<24}{'mean diff':>10}{'A wins':>8}{'Wilcoxon p':>12}   hierarchical 95% CI")
    verdict = {}
    for A, B in key:
        ra = np.array([e['ROC'] for e in res[A]]); rb = np.array([e['ROC'] for e in res[B]]); d = ra - rb
        try:
            p = wilcoxon(ra, rb).pvalue if np.any(d != 0) else 1.0
        except ValueError:
            p = 1.0
        lo, hi = hier_boot(Z[A], Z[B], oods, fams, a.n_boot)
        verdict[B] = 'A better' if lo > 0 else ('B better' if hi < 0 else 'no clear difference')
        log(f'  {A:<22}{B:<24}{d.mean():>+10.3f}{int((d > 0).sum()):>5}/{len(d):<3}{p:>11.4f}   [{lo:+.3f}, {hi:+.3f}]  -> {verdict[B]}')
    cpairs = [n for n in names if n.startswith('C:')]
    if cpairs:
        best_c = max(cpairs, key=lambda n: np.mean([e['ROC'] for e in res[n]]))
        ok = verdict.get('amp-64 alone') == 'A better' and verdict.get(best_c) == 'A better'
        log(f'\n  PROOF-OF-CONCEPT CRITERION (fixed in advance): beats amp-64 alone AND the best classical cross pair ({best_c})')
        log(f'  -> {"MET" if ok else "NOT met"}  (vs amp-64: {verdict.get("amp-64 alone")}; vs {best_c}: {verdict.get(best_c)})')
    # ---------------- weight curve (descriptive)
    W = np.round(np.arange(0, 1 + 1e-9, a.curve_step), 3); curve = []
    pairs = [(Q, 'angle-8', 'amp-64')]
    if 'C: enc-64 + enc-8' in ens:
        pairs.append(('C: enc-64 + enc-8', 'enc-8', 'enc-64'))
    zsingle = {}
    for m in {m for _, x, y in pairs for m in (x, y)}:
        zsingle[m] = []
    for (r, s) in tasks:
        for m in zsingle:
            zsingle[m].append(zscore(load_task(F[m], r, s)))
    log('\nWEIGHT SENSITIVITY CURVE -- w * z_first + (1 - w) * z_second on the TEST sets; DESCRIPTIVE ONLY (no weight selected)')
    for title, first, second in pairs:
        log(f'\n  {title}: w * {first} + (1 - w) * {second}')
        log(f"  {'w':>6}" + ''.join(f'{c:>9}' for c in cols)); fam_roc = {}
        for w in W:
            ev = [evaluate(w * z1 + (1 - w) * z2, o) for z1, z2, o in zip(zsingle[first], zsingle[second], oods)]
            mark = '   <- mean' if abs(w - 0.5) < 1e-9 else (f'   <- {second} alone' if w == 0 else (f'   <- {first} alone' if w == 1 else ''))
            log(f'  {w:>6.1f}' + ''.join(f'{np.mean([e[c] for e in ev]):>9.3f}' for c in cols) + mark)
            for (r, s), f_, e in zip(tasks, fams, ev):
                curve.append({'pair': title, 'w': float(w), 'run': r, 'split': s, 'family': f_, **{c: e[c] for c in cols}})
            for f_ in fl:
                fam_roc.setdefault(f_, []).append(np.mean([e['ROC'] for e, ff in zip(ev, fams) if ff == f_]))
        log(f"  per family, raw ROC-AUC by w:\n  {'family':<14}" + ''.join(f'{w:>7.1f}' for w in W) + '   best w')
        for f_ in fl:
            v = fam_roc[f_]; log(f'  {f_:<14}' + ''.join(f'{x:>7.3f}' for x in v) + f'   {W[int(np.argmax(v))]:>5.1f}')
    out = os.path.join(R64, 'comparison'); os.makedirs(os.path.join(out, 'plots'), exist_ok=True)
    for (name, v) in res.items():
        for (r, s), f, e in zip(tasks, fams, v):
            rows.append({'ensemble': name, 'run': r, 'split': s, 'family': f, **{c: e[c] for c in cols}})
    open(os.path.join(out, 'ensemble_cross.txt'), 'w').write('\n'.join(lines) + '\n')
    for fn, data, fields in (('ensemble_cross_per_task.csv', rows, ['ensemble', 'run', 'split', 'family'] + cols),
                             ('ensemble_cross_curve.csv', curve, ['pair', 'w', 'run', 'split', 'family'] + cols)):
        with open(os.path.join(out, fn), 'w', newline='') as fh:
            wr = csv.DictWriter(fh, fieldnames=fields); wr.writeheader(); wr.writerows(data)
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(pairs), figsize=(6.5 * len(pairs), 4.5), squeeze=False)
        for ax, (title, first, second) in zip(axes[0], pairs):
            pr = [c for c in curve if c['pair'] == title]
            for f_ in fl:
                ax.plot(W, [np.mean([c['ROC'] for c in pr if c['w'] == w and c['family'] == f_]) for w in W], lw=1, alpha=0.6, label=f_)
            ax.plot(W, [np.mean([c['ROC'] for c in pr if c['w'] == w]) for w in W], 'k-o', lw=2.5, label='mean')
            ax.axvline(0.5, color='grey', ls='--', lw=1); ax.set_title(title, fontsize=10)
            ax.set_xlabel(f'w  (0 = {second} alone, 1 = {first} alone)'); ax.set_ylabel('raw ROC-AUC (test)')
        axes[0][-1].legend(fontsize=7, loc='center left', bbox_to_anchor=(1.0, 0.5))
        fig.suptitle('Cross-input weight sensitivity (descriptive)', fontsize=10); fig.tight_layout()
        fig.savefig(os.path.join(out, 'plots', 'ensemble_cross_curve.png'), dpi=150, bbox_inches='tight'); plt.close(fig)
    except Exception as ex:
        log(f'  (plot skipped: {ex})')
    print(f'\nSaved: {out}/ensemble_cross.txt, ensemble_cross_per_task.csv, ensemble_cross_curve.csv, plots/ensemble_cross_curve.png')


if __name__ == '__main__':
    main()
