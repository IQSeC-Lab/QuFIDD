#!/usr/bin/env python3
"""
ensemble_angle_amp.py -- capacity_matched / capacity_matched_8fam (top level, next to evaluate.py)
======================================================================================================================
Score-level ensemble of the two quantum encodings (angle + amplitude) from SAVED scores -- nothing is retrained.

RULE (fixed before looking at results): each model's raw distance is standardized by that model's own TRAINING raw
scores, z = (d - median_train) / (1.4826 MAD_train) (signed: only farther than usual counts); the ensemble score is the
MEAN of the two z-scores (primary). The MAX of the two is reported as a secondary rule.

ENSEMBLES (all with the same rule):
  Q   angle + amp                  the quantum ensemble: angle-8 and amp-8 of the SAME seed on the same split
                                   (192 + 192 = 384 params at PCA-8)
  Qc  amp + amp, angle + angle     optional (--seed-controls): SEED ensembles of one encoding (same split, the next
                                   seed), to separate encoding complementarity from plain ensembling
  C   CADE enc + tuned, enc + full, tuned + full    classical ensembles of comparable size
Components alone are included for reference.

METRICS: ROC-AUC, PR-AUC (raw, threshold-free) and, at the best cut-off of the ranking by the ensemble score
(T = 0, analyst inspects the top k): F1, precision, recall, FPR, FNR, effort (k / number of drifting samples).
WEIGHT CURVE: score = w * z_angle + (1 - w) * z_amp for w = 0, 0.1, ..., 1 (and the same for CADE enc + tuned), on the
test sets -- DESCRIPTIVE ONLY, no weight is selected (a test-chosen weight would be test-set selection).
STATISTICS: per-task Wilcoxon and a hierarchical bootstrap (families -> seeds -> test samples) of the ROC-AUC
difference for the key comparisons.

Usage (from capacity_matched/ or capacity_matched_8fam/):
  python3 ensemble_angle_amp.py pca8                     # default n_boot 500
  python3 ensemble_angle_amp.py pca8 --n-boot 200
  python3 ensemble_angle_amp.py pca8 --seed-controls     # also the amp + amp / angle + angle seed ensembles
Output: <pca dir>/comparison/ensemble_angle_amp.txt, ensemble_per_task.csv
"""
import os, glob, json, csv, argparse
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import wilcoxon

ROLES = {'angle': 'quantum_angle', 'amp': 'quantum_amp', 'enc': 'cade_enc', 'tuned': 'cade_tuned', 'full': 'cade_full'}


def find(root, prefix):
    m = sorted(f for f in os.listdir(root) if f.startswith(prefix) and os.path.isdir(os.path.join(root, f, 'results')))
    return m[0] if m else None


def load_task(root, folder, run, split):
    d = os.path.join(root, folder, 'results', f'run{run}', f'split{split}')
    if not os.path.exists(os.path.join(d, 'ood_scores.csv')):
        return None
    c = np.genfromtxt(os.path.join(d, 'ood_scores.csv'), delimiter=',', names=True)
    tr = np.load(os.path.join(d, 'train_raw_scores.npy'))
    fam = json.load(open(os.path.join(d, 'summary.json'))).get('ood_family', f'split{split}')
    return {'raw': c['raw_score'], 'ood': c['is_ood'].astype(int), 'label': c['true_label'], 'train': tr, 'family': fam}


def zscore(t):
    med = np.median(t['train']); mad = 1.4826 * np.median(np.abs(t['train'] - med))
    return (t['raw'] - med) / (mad if mad > 0 else 1e-12)


def ranked(score, ood):
    """Best cut-off of the ranking (T = 0): F1, P, R, FPR, FNR, effort."""
    o = np.argsort(-score); y = ood[o]; tp = np.cumsum(y); k = np.arange(1, len(y) + 1)
    n_pos, n_neg = int(ood.sum()), int(len(ood) - ood.sum())
    P, R = tp / k, tp / max(n_pos, 1); F = np.where(P + R > 0, 2 * P * R / (P + R + 1e-12), 0)
    i = int(np.argmax(F)); fp = (k[i] - tp[i])
    return {'F1': F[i], 'P': P[i], 'R': R[i], 'FPR': fp / max(n_neg, 1), 'FNR': 1 - R[i], 'Effort': k[i] / max(n_pos, 1)}


def evaluate(score, ood):
    e = {'ROC': roc_auc_score(ood, score), 'PR': average_precision_score(ood, score)}
    e.update(ranked(score, ood)); return e


def hier_boot(scoresA, scoresB, oods, fams, n_boot, seed=0):
    """Hierarchical bootstrap of mean ROC difference: families -> seeds (tasks of that family) -> test samples."""
    rng = np.random.default_rng(seed); by_fam = {}
    for i, f in enumerate(fams):
        by_fam.setdefault(f, []).append(i)
    keys = list(by_fam); diffs = []
    for _ in range(n_boot):
        vals = []
        for f in rng.choice(keys, len(keys)):
            for i in rng.choice(by_fam[f], len(by_fam[f])):
                n = len(oods[i]); idx = rng.integers(0, n, n); o = oods[i][idx]
                if o.min() == o.max():
                    continue
                vals.append(roc_auc_score(o, scoresA[i][idx]) - roc_auc_score(o, scoresB[i][idx]))
        diffs.append(np.mean(vals))
    return float(np.mean(diffs)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('pca_dir'); ap.add_argument('--n-boot', type=int, default=500)
    ap.add_argument('--curve-step', type=float, default=0.1, help='step of the weight sensitivity curve (default 0.1)')
    ap.add_argument('--seed-controls', action='store_true',
                    help='also build amp + amp and angle + angle ensembles of two different seeds (off by default)')
    a = ap.parse_args(); root = os.path.abspath(a.pca_dir)
    F = {r: find(root, p) for r, p in ROLES.items()}
    missing = [r for r in ('angle', 'amp') if F[r] is None]
    if missing:
        raise SystemExit(f'Missing quantum folders for: {missing}')
    runs = sorted({int(p.split('run')[-1]) for p in glob.glob(os.path.join(root, F['amp'], 'results', 'run*'))})
    splits = sorted({int(p.split('split')[-1]) for p in glob.glob(os.path.join(root, F['amp'], 'results', 'run*', 'split*'))})
    nr = len(runs)
    ens = {'amp alone': [('amp', 0)], 'angle alone': [('angle', 0)],
           'Q: angle + amp': [('angle', 0), ('amp', 0)],
           'Qc: amp + amp (seeds)': [('amp', 0), ('amp', 1)], 'Qc: angle + angle (seeds)': [('angle', 0), ('angle', 1)]}
    if nr < 2 or not a.seed_controls:                           # seed ensembles only on request (need 2 seeds)
        for k in [k for k in ens if k.startswith('Qc:')]:
            del ens[k]
    for r in ('enc', 'tuned', 'full'):
        if F[r]:
            ens[f'{r} alone'] = [(r, 0)]
    for x, y in (('enc', 'tuned'), ('enc', 'full'), ('tuned', 'full')):
        if F[x] and F[y]:
            ens[f'C: {x} + {y}'] = [(x, 0), (y, 0)]
    lines, rows = [], []
    def log(s=''):
        print(s, flush=True); lines.append(s)
    log('=' * 118); log(f'ANGLE + AMPLITUDE ENSEMBLE -- {root}'); log('=' * 118)
    log('  folders: ' + ', '.join(f'{r}={F[r]}' for r in F if F[r]))
    log('  rule: mean of the members\' z-scores, z = (d - median_train) / (1.4826 MAD_train), per model (primary); max (secondary)')
    per = {(name, rule): [] for name in ens for rule in ('mean', 'max')}; oods, fams, tasks = [], [], []
    for s in splits:
        for r in runs:
            base = load_task(root, F['amp'], r, s)
            if base is None:
                continue
            ok = True; zs = {}
            for name, members in ens.items():
                mz = []
                for role, shift in members:
                    t = load_task(root, F[role], runs[(runs.index(r) + shift) % nr], s)
                    if t is None or len(t['ood']) != len(base['ood']) or not np.array_equal(t['label'], base['label']):
                        ok = False; break
                    mz.append(zscore(t))
                if not ok:
                    break
                zs[name] = np.array(mz)
            if not ok:
                log(f'  run{r} split{s}: a member is missing or its test set differs -- task skipped'); continue
            oods.append(base['ood']); fams.append(base['family']); tasks.append((r, s))
            for name, Z in zs.items():
                for rule, sc in (('mean', Z.mean(0)), ('max', Z.max(0))):
                    per[(name, rule)].append(sc)
    log(f'  tasks: {len(tasks)} ({len(set(fams))} families x {nr} seeds); seed ensembles pair seed i with seed i+1 on the same split'
        + ('' if (nr > 1 and a.seed_controls) else '  -- seed ensembles not built (use --seed-controls)'))
    res = {k: [evaluate(sc, o) for sc, o in zip(v, oods)] for k, v in per.items()}
    cols = ['ROC', 'PR', 'F1', 'P', 'R', 'FPR', 'FNR', 'Effort']
    for rule in ('mean', 'max'):
        log(f'\nOVERALL -- rule: {rule} of z-scores (single models: their own z-score)   mean over {len(tasks)} tasks')
        log(f"  {'model / ensemble':<30}" + ''.join(f'{c:>9}' for c in cols))
        for name in ens:
            if rule == 'max' and len(ens[name]) == 1:
                continue
            v = res[(name, rule)]
            log(f'  {name:<30}' + ''.join(f'{np.mean([e[c] for e in v]):>9.3f}' for c in cols))
    log('\nPER FAMILY -- raw ROC-AUC, rule: mean (mean over seeds)')
    names = list(ens); fl = sorted(set(fams))
    log(f"  {'family':<14}" + ''.join(f'{n[:15]:>16}' for n in names))
    for f in fl:
        idx = [i for i, x in enumerate(fams) if x == f]
        log(f'  {f:<14}' + ''.join(f'{np.mean([res[(n, "mean")][i]["ROC"] for i in idx]):>16.3f}' for n in names))
    key = [('Q: angle + amp', 'amp alone'), ('Q: angle + amp', 'angle alone')]
    key += [('Q: angle + amp', n) for n in names if n.startswith('Qc:')]
    key += [('Q: angle + amp', n) for n in names if n.startswith('C:') or n in ('enc alone', 'tuned alone')]
    log(f'\nKEY COMPARISONS -- raw ROC-AUC, rule: mean (A minus B); hierarchical bootstrap n = {a.n_boot}')
    log(f"  {'A':<18}{'B':<28}{'mean diff':>10}{'A wins':>8}{'Wilcoxon p':>12}   hierarchical 95% CI")
    for A, B in key:
        ra = np.array([e['ROC'] for e in res[(A, 'mean')]]); rb = np.array([e['ROC'] for e in res[(B, 'mean')]])
        d = ra - rb
        try:
            p = wilcoxon(ra, rb).pvalue if np.any(d != 0) else 1.0
        except ValueError:
            p = 1.0
        m, lo, hi = hier_boot(per[(A, 'mean')], per[(B, 'mean')], oods, fams, a.n_boot)
        verdict = 'A better' if lo > 0 else ('B better' if hi < 0 else 'no clear difference')
        log(f'  {A:<18}{B:<28}{d.mean():>+10.3f}{int((d > 0).sum()):>5}/{len(d):<3}{p:>11.4f}   [{lo:+.3f}, {hi:+.3f}]  -> {verdict}')
    # ------------------------------------------------------------------ weight sensitivity curve (descriptive only)
    W = np.round(np.arange(0.0, 1.0 + 1e-9, a.curve_step), 3); curve_rows = []
    pairs = [('Q: w * angle + (1 - w) * amp', 'angle alone', 'amp alone')]
    if ('enc alone', 'mean') in per and ('tuned alone', 'mean') in per:
        pairs.append(('C: w * enc + (1 - w) * tuned', 'enc alone', 'tuned alone'))
    log('\nWEIGHT SENSITIVITY CURVE -- score = w * z_first + (1 - w) * z_second, every w on the TEST sets.')
    log('  DESCRIPTIVE ONLY: no weight is selected here. w = 0.5 is the ensemble above; w = 0 and w = 1 are the two')
    log('  members alone. A peak away from 0.5 is chosen with test labels and must not be reported as a model result.')
    for title, first, second in pairs:
        log(f'\n  {title}   (mean over {len(tasks)} tasks)')
        log(f"  {'w':>6}" + ''.join(f'{c:>9}' for c in cols))
        fam_roc = {}
        for w in W:
            ev = [evaluate(w * z1 + (1 - w) * z2, o) for z1, z2, o in zip(per[(first, 'mean')], per[(second, 'mean')], oods)]
            mark = '   <- mean ensemble' if abs(w - 0.5) < 1e-9 else ('   <- ' + second.replace(' alone', '') + ' alone' if w == 0
                                                                      else ('   <- ' + first.replace(' alone', '') + ' alone' if w == 1 else ''))
            log(f'  {w:>6.1f}' + ''.join(f'{np.mean([e[c] for e in ev]):>9.3f}' for c in cols) + mark)
            for (r, s_), f_, e in zip(tasks, fams, ev):
                curve_rows.append({'pair': title, 'w': float(w), 'run': r, 'split': s_, 'family': f_, **{c: e[c] for c in cols}})
            for f_ in fl:
                fam_roc.setdefault(f_, []).append(np.mean([e['ROC'] for e, ff in zip(ev, fams) if ff == f_]))
        log(f'  per family, raw ROC-AUC by w:')
        log(f"  {'family':<14}" + ''.join(f'{w:>7.1f}' for w in W) + '   best w')
        for f_ in fl:
            v = fam_roc[f_]
            log(f'  {f_:<14}' + ''.join(f'{x:>7.3f}' for x in v) + f'   {W[int(np.argmax(v))]:>5.1f}')
    with open(os.path.join(root, 'comparison', 'ensemble_weight_curve.csv') if os.path.isdir(os.path.join(root, 'comparison'))
              else os.path.join(root, 'ensemble_weight_curve.csv'), 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=['pair', 'w', 'run', 'split', 'family'] + cols); wr.writeheader(); wr.writerows(curve_rows)
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(pairs), figsize=(6.5 * len(pairs), 4.5), squeeze=False)
        for ax, (title, first, second) in zip(axes[0], pairs):
            pr = [r for r in curve_rows if r['pair'] == title]
            for f_ in fl:
                ax.plot(W, [np.mean([r['ROC'] for r in pr if r['w'] == w and r['family'] == f_]) for w in W], lw=1, alpha=0.6, label=f_)
            ax.plot(W, [np.mean([r['ROC'] for r in pr if r['w'] == w]) for w in W], 'k-o', lw=2.5, label='mean')
            ax.axvline(0.5, color='grey', ls='--', lw=1)
            ax.set_xlabel(f'w  (0 = {second.replace(" alone", "")} alone, 1 = {first.replace(" alone", "")} alone)')
            ax.set_ylabel('raw ROC-AUC (test)'); ax.set_title(title, fontsize=10)
        axes[0][-1].legend(fontsize=7, loc='center left', bbox_to_anchor=(1.0, 0.5))
        fig.suptitle('Weight sensitivity (descriptive; no weight selected on test data)', fontsize=10)
        fig.tight_layout(); os.makedirs(os.path.join(root, 'comparison', 'plots'), exist_ok=True)
        fig.savefig(os.path.join(root, 'comparison', 'plots', 'ensemble_weight_curve.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
    except Exception as ex:
        log(f'  (plot skipped: {ex})')
    for (name, rule), v in res.items():
        for (r, s), f, e in zip(tasks, fams, v):
            rows.append({'ensemble': name, 'rule': rule, 'run': r, 'split': s, 'family': f, **{c: e[c] for c in cols}})
    out = os.path.join(root, 'comparison'); os.makedirs(out, exist_ok=True)
    open(os.path.join(out, 'ensemble_angle_amp.txt'), 'w').write('\n'.join(lines) + '\n')
    with open(os.path.join(out, 'ensemble_per_task.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['ensemble', 'rule', 'run', 'split', 'family'] + cols); w.writeheader(); w.writerows(rows)
    print(f'\nSaved: {out}/ensemble_angle_amp.txt, ensemble_per_task.csv, ensemble_weight_curve.csv, plots/ensemble_weight_curve.png')


if __name__ == '__main__':
    main()
