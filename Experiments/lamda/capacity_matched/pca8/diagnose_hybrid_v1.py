#!/usr/bin/env python3
"""
diagnose_hybrid_v1.py -- pca8 (run from capacity_matched/pca8/ or capacity_matched_8fam/pca8/)
======================================================================================================================
Re-scores Quantum hybrid v1 from its SAVED states (results/run*/split*/states.npz) -- nothing is retrained -- to find
out why it behaves like amp-8 instead of combining amp-8 and angle-8. Each view is scored separately:

  A pure          amplitude register alone, pure prototypes                (what amp-8 does)
  B pure          angle register alone, pure prototypes                    (what the hybrid uses for that register)
  B mixed         angle register alone, MIXED prototypes rho_c = mean |psi><psi|   (what angle-8 does)
  A x B pure      the hybrid's own score (product, pure prototypes)        -- should reproduce its saved results
  A x B (B mixed) product with the angle register scored like angle-8

For each view: raw-distance ROC-AUC, MAD-score ROC-AUC and ranked F1 at T = 0, overall and per family, next to the
saved angle-8 and amp-8 results for the same tasks. Also: Spearman correlation of the product's test score with each
register's score (which view dominates the product?) and the spread of each register's fidelities.

Hypotheses tested:
  H1  the angle register is too weak  -> B (pure and mixed) is far below angle-8 on droidkungfu
  H2  the prototype type matters      -> B mixed is much better than B pure on droidkungfu
  H3  the amplitude view dominates     -> the product's score correlates far more with A than with B

Usage (from pca8/):  python3 diagnose_hybrid_v1.py
Output: diagnosis_hybrid_v1.txt and diagnosis_hybrid_v1.csv in pca8/comparison/
"""
import os, glob, json, csv
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve

HERE = os.path.dirname(os.path.abspath(__file__))
HYB = os.path.join(HERE, 'quantum_hybrid_v1', 'results')
REF = {'angle-8 (saved)': os.path.join(HERE, 'quantum_angle_8', 'results'),
       'amp-8 (saved)': os.path.join(HERE, 'quantum_amp_8', 'results')}
VIEWS = ['A pure', 'B pure', 'B mixed', 'A x B pure', 'A x B (B mixed)', 'fused mean-z', 'fused max-z']
FUSED = ['fused mean-z', 'fused max-z']


def mad_stats(d):
    m = float(np.median(d)); s = 1.4826 * float(np.median(np.abs(d - m)))
    return m, (s if s > 0 else 1e-12)


def pure_protos(S, y, C):
    V = []
    for c in range(C):
        P = S[y == c]; rho = P.T @ P.conj() / len(P)
        V.append(np.linalg.eigh(rho)[1][:, -1].conj())
    return np.stack(V)


def fid_pure(S, V):
    return np.abs(S @ V.T) ** 2                                       # [N, C]


def fid_mixed(S, Str, ytr, C, loo=False):
    """<psi|rho_c|psi> = mean_j |<psi|psi_j>|^2 over class c (leave-one-out for training samples, as angle-8)."""
    F = np.zeros((len(S), C))
    for c in range(C):
        Fk = np.abs(S.conj() @ Str[ytr == c].T) ** 2; n = Fk.shape[1]
        if loo:
            own = ytr == c
            F[:, c] = Fk.mean(1); F[own, c] = (Fk[own].sum(1) - 1.0) / max(n - 1, 1)
        else:
            F[:, c] = Fk.mean(1)
    return F


def scores(Ftr, Fte, ytr):
    """Nearest-prototype raw distance and per-class MAD score from per-class fidelities."""
    Dtr, Dte = 1 - Ftr, 1 - Fte; C = Ftr.shape[1]; A = []
    for c in range(C):
        m, s = mad_stats(Dtr[ytr == c, c]); A.append(np.abs(Dte[:, c] - m) / s)
    return Dte.min(1), np.min(A, axis=0)


def zmat(Ftr, Fte, ytr):
    """Per-class SIGNED MAD-standardized distances (d_c - median_c) / (1.4826 MAD_c), statistics from TRAINING only.
    Signed, not absolute: being CLOSER than usual to a family must not count as unusual when the views are fused."""
    Dtr, Dte = 1 - Ftr, 1 - Fte; Z = np.zeros_like(Dte)
    for c in range(Ftr.shape[1]):
        m, s = mad_stats(Dtr[ytr == c, c]); Z[:, c] = (Dte[:, c] - m) / s
    return Z


def ranked_f1(raw, ood):
    p, r, _ = precision_recall_curve(ood, raw)
    return float(np.max(np.where(p + r > 0, 2 * p * r / (p + r + 1e-12), 0.0)))


def evaluate(raw, mad, ood):
    return {'ROC_raw': roc_auc_score(ood, raw), 'ROC_mad': roc_auc_score(ood, mad), 'F1_T0': ranked_f1(raw, ood)}


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    tasks = sorted(glob.glob(os.path.join(HYB, 'run*', 'split*', 'states.npz')))
    if not tasks:
        raise SystemExit(f'No hybrid v1 states found under {HYB}')
    rows, lines = [], []
    def log(s=''):
        print(s, flush=True); lines.append(s)
    log('=' * 112); log(f'HYBRID v1 DIAGNOSIS -- {len(tasks)} tasks, saved states re-scored (no retraining)'); log('=' * 112)
    for f in tasks:
        d = os.path.dirname(f); z = np.load(f)
        A_tr, A_te, B_tr, B_te = z['states_train_r0'], z['states_test_r0'], z['states_train_r1'], z['states_test_r1']
        ytr = z['y_train']; C = len(np.unique(ytr))
        ood = np.genfromtxt(os.path.join(d, 'ood_scores.csv'), delimiter=',', names=True)['is_ood'].astype(int)
        fam = json.load(open(os.path.join(d, 'summary.json')))['ood_family']
        VA, VB = pure_protos(A_tr, ytr, C), pure_protos(B_tr, ytr, C)
        FA_tr, FA_te = fid_pure(A_tr, VA), fid_pure(A_te, VA)
        FBp_tr, FBp_te = fid_pure(B_tr, VB), fid_pure(B_te, VB)
        FBm_tr, FBm_te = fid_mixed(B_tr, B_tr, ytr, C, loo=True), fid_mixed(B_te, B_tr, ytr, C)
        views = {'A pure': (FA_tr, FA_te), 'B pure': (FBp_tr, FBp_te), 'B mixed': (FBm_tr, FBm_te),
                 'A x B pure': (FA_tr * FBp_tr, FA_te * FBp_te), 'A x B (B mixed)': (FA_tr * FBm_tr, FA_te * FBm_te)}
        rel = d.split(os.sep + 'results' + os.sep)[-1]
        row = {'task': rel, 'family': fam}; raws = {}
        for v, (Ftr, Fte) in views.items():
            raw, mad = scores(Ftr, Fte, ytr); raws[v] = raw
            for k, val in evaluate(raw, mad, ood).items():
                row[f'{v} | {k}'] = val
        # balanced fusion of the two registers (each standardized by its OWN training MAD, per family):
        #   mean-z: family score = mean of the two registers' standardized distances, sample score = min over families
        #   max-z : family score = max of the two (unusual in EITHER view),           sample score = min over families
        ZA, ZB = zmat(FA_tr, FA_te, ytr), zmat(FBp_tr, FBp_te, ytr)
        for v, sc in (('fused mean-z', ((ZA + ZB) / 2).min(1)), ('fused max-z', np.maximum(ZA, ZB).min(1))):
            row[f'{v} | ROC_raw'] = roc_auc_score(ood, sc); row[f'{v} | ROC_mad'] = row[f'{v} | ROC_raw']
            row[f'{v} | F1_T0'] = ranked_f1(sc, ood)
        row['corr(product, A)'] = spearman(raws['A x B pure'], raws['A pure'])
        row['corr(product, B)'] = spearman(raws['A x B pure'], raws['B pure'])
        nn = lambda F: F.max(1)                                        # fidelity to the nearest prototype
        row['spread A (std nearest F)'] = float(np.std(nn(FA_te))); row['spread B (std nearest F)'] = float(np.std(nn(FBp_te)))
        row['mean A nearest F'] = float(np.mean(nn(FA_te))); row['mean B nearest F'] = float(np.mean(nn(FBp_te)))
        for name, base in REF.items():
            p = os.path.join(base, rel, 'summary.json')
            if os.path.exists(p):
                e = json.load(open(p))['evaluation']
                row[f'{name} | ROC_raw'] = e['threshold_free']['ROC_AUC']; row[f'{name} | ROC_mad'] = e['ROC_AUC_mad']
                row[f'{name} | F1_T0'] = e['threshold_free']['F1']
        saved = json.load(open(os.path.join(d, 'summary.json')))['evaluation']['threshold_free']['ROC_AUC']
        row['check: saved hybrid ROC_raw'] = saved
        rows.append(row)
    fams = []
    for r in rows:
        if r['family'] not in fams:
            fams.append(r['family'])
    cols = VIEWS + [n for n in REF if any(f'{n} | ROC_raw' in r for r in rows)]
    for metric in ['ROC_raw', 'ROC_mad', 'F1_T0']:
        log(f'\n{metric} (mean over seeds)')
        log(f"  {'family':<14}" + ''.join(f'{c:>18}' for c in cols))
        for fam in fams + ['ALL']:
            sel = rows if fam == 'ALL' else [r for r in rows if r['family'] == fam]
            vals = [np.mean([r.get(f'{c} | {metric}', np.nan) for r in sel]) for c in cols]
            log(f'  {fam:<14}' + ''.join(f'{v:>18.3f}' for v in vals))
    log('\nWhich register drives the product? (Spearman correlation of test raw scores; fidelity spread on test)')
    log(f"  {'family':<14}{'corr(prod,A)':>14}{'corr(prod,B)':>14}{'mean F_A':>10}{'std F_A':>10}{'mean F_B':>10}{'std F_B':>10}")
    for fam in fams + ['ALL']:
        sel = rows if fam == 'ALL' else [r for r in rows if r['family'] == fam]
        m = lambda k: np.mean([r[k] for r in sel])
        log(f'  {fam:<14}{m("corr(product, A)"):>14.3f}{m("corr(product, B)"):>14.3f}{m("mean A nearest F"):>10.3f}'
            f'{m("spread A (std nearest F)"):>10.3f}{m("mean B nearest F"):>10.3f}{m("spread B (std nearest F)"):>10.3f}')
    dev = max(abs(r['A x B pure | ROC_raw'] - r['check: saved hybrid ROC_raw']) for r in rows)
    log(f'\nConsistency check: max |re-scored product ROC_raw - saved hybrid ROC_raw| = {dev:.4f} (should be ~0)')
    log('\nFused views: each register standardized by its own training median/MAD per family (signed: only FARTHER than')
    log('usual counts), then mean or max of the two registers, min over families;')
    log('a fused view has ONE score, so its ROC_raw and ROC_mad columns are identical and F1_T0 ranks by that score.')
    log('These are FEASIBILITY checks (fusion chosen after training), not results to report.')
    log('\nReading:  H1 (weak angle register): "B mixed"/"B pure" far below "angle-8 (saved)" on the angle-8 families.')
    log('          H2 (prototype type): "B mixed" clearly above "B pure".')
    log('          H3 (amplitude dominates): corr(prod, A) >> corr(prod, B) and/or std F_A >> std F_B.')
    out = os.path.join(HERE, 'comparison'); os.makedirs(out, exist_ok=True)
    open(os.path.join(out, 'diagnosis_hybrid_v1.txt'), 'w').write('\n'.join(lines) + '\n')
    keys = list(rows[0].keys())
    with open(os.path.join(out, 'diagnosis_hybrid_v1.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys, restval=''); w.writeheader(); w.writerows(rows)
    print(f'\nSaved: {out}/diagnosis_hybrid_v1.txt and .csv')


if __name__ == '__main__':
    main()
