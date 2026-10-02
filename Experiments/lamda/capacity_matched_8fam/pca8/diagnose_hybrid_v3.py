#!/usr/bin/env python3
"""
diagnose_hybrid_v3.py -- pca8 (run from capacity_matched/pca8/ or capacity_matched_8fam/pca8/)
======================================================================================================================
Re-scores Quantum hybrid v3 from its SAVED states (results/run*/split*/states.npz) -- nothing is retrained -- to test
whether weighting the two registers by their RELIABILITY improves on v3's equal-weight (mean-z) fusion.

Per register r (A = amplitude, B = angle) and known family c, with pure-state prototypes and training statistics only:
  z_rc(x)        = (d_rc(x) - median_rc) / (1.4826 MAD_rc)        signed standardized distance to family c
  reliability_rc = median of z_rc over the TRAINING samples of the OTHER known families
                   (how far the other known families sit from family c in register r, in units of c's own spread:
                   a register that separates c well from the known families gets a high value)
  weight w_rc    = max(reliability_rc, 0) / sum_r max(reliability_r'c, 0)     (equal weights if both are <= 0)

Scores compared (sample score = min over families of the family score):
  A only / B only      one register alone
  equal (v3)           s_c = (z_Ac + z_Bc) / 2                    -- must reproduce v3's saved results (consistency check)
  reliability-weighted s_c = w_Ac z_Ac + w_Bc z_Bc                -- the PRE-SPECIFIED candidate for "hybrid v4"
  best register        s_c = z_rc of the register with the higher reliability for c (hard choice; sensitivity check)
The weights never use the held-out (drift) family or any test label. Saved angle-8, amp-8 and hybrid v2 results for
the same tasks are shown for reference.

Usage (from pca8/):  python3 diagnose_hybrid_v3.py
Output: comparison/diagnosis_hybrid_v3.txt and .csv
"""
import os, glob, json, csv
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve

HERE = os.path.dirname(os.path.abspath(__file__))
HYB = os.path.join(HERE, 'quantum_hybrid_v3', 'results')
REF = {'angle-8 (saved)': 'quantum_angle_8', 'amp-8 (saved)': 'quantum_amp_8', 'hybrid v2 (saved)': 'quantum_hybrid_v2'}
VIEWS = ['A only', 'B only', 'equal (v3)', 'reliability-weighted', 'best register']


def mad_stats(d):
    m = float(np.median(d)); s = 1.4826 * float(np.median(np.abs(d - m)))
    return m, (s if s > 0 else 1e-12)


def pure_protos(S, y, C):
    V = []
    for c in range(C):
        P = S[y == c]; V.append(np.linalg.eigh(P.T @ P.conj() / len(P))[1][:, -1].conj())
    return np.stack(V)


def zscores(Str, Ste, ytr, C):
    """Signed standardized distances for training and test samples, statistics from each family's training samples."""
    V = pure_protos(Str, ytr, C)
    Dtr, Dte = 1 - np.abs(Str @ V.T) ** 2, 1 - np.abs(Ste @ V.T) ** 2
    Ztr, Zte = np.zeros_like(Dtr), np.zeros_like(Dte)
    for c in range(C):
        m, s = mad_stats(Dtr[ytr == c, c]); Ztr[:, c] = (Dtr[:, c] - m) / s; Zte[:, c] = (Dte[:, c] - m) / s
    return Ztr, Zte


def ranked_f1(score, ood):
    p, r, _ = precision_recall_curve(ood, score)
    return float(np.max(np.where(p + r > 0, 2 * p * r / (p + r + 1e-12), 0.0)))


def main():
    tasks = sorted(glob.glob(os.path.join(HYB, 'run*', 'split*', 'states.npz')))
    if not tasks:
        raise SystemExit(f'No hybrid v3 states found under {HYB}')
    rows, lines = [], []
    def log(s=''):
        print(s, flush=True); lines.append(s)
    log('=' * 112); log(f'HYBRID v3 RELIABILITY-WEIGHTING DIAGNOSIS -- {len(tasks)} tasks (saved states, no retraining)'); log('=' * 112)
    for f in tasks:
        d = os.path.dirname(f); z = np.load(f); ytr = z['y_train']; C = len(np.unique(ytr))
        ood = np.genfromtxt(os.path.join(d, 'ood_scores.csv'), delimiter=',', names=True)['is_ood'].astype(int)
        summ = json.load(open(os.path.join(d, 'summary.json'))); fam = summ['ood_family']
        ZA_tr, ZA_te = zscores(z['states_train_r0'], z['states_test_r0'], ytr, C)
        ZB_tr, ZB_te = zscores(z['states_train_r1'], z['states_test_r1'], ytr, C)
        rel = np.zeros((2, C))
        for c in range(C):
            other = ytr != c
            rel[0, c] = np.median(ZA_tr[other, c]); rel[1, c] = np.median(ZB_tr[other, c])
        pos = np.clip(rel, 0, None); tot = pos.sum(0)
        W = np.where(tot > 0, pos / np.where(tot > 0, tot, 1), 0.5)            # [2, C]
        best = (rel[1] > rel[0]).astype(float)                                   # 1 -> register B for family c
        fused = {'A only': ZA_te, 'B only': ZB_te, 'equal (v3)': (ZA_te + ZB_te) / 2,
                 'reliability-weighted': W[0] * ZA_te + W[1] * ZB_te,
                 'best register': (1 - best) * ZA_te + best * ZB_te}
        rlt = os.path.relpath(d, HYB); row = {'task': rlt, 'family': fam,
                                              'mean weight on B (angle)': float(W[1].mean()),
                                              'families choosing B': int(best.sum()), 'known families': C}
        for v, F in fused.items():
            sc = F.min(1); row[f'{v} | ROC'] = roc_auc_score(ood, sc); row[f'{v} | F1_T0'] = ranked_f1(sc, ood)
        row['check: saved v3 ROC_raw'] = summ['evaluation']['threshold_free']['ROC_AUC']
        for name, folder in REF.items():
            p = os.path.join(HERE, folder, 'results', rlt, 'summary.json')
            if os.path.exists(p):
                e = json.load(open(p))['evaluation']['threshold_free']
                row[f'{name} | ROC'] = e['ROC_AUC']; row[f'{name} | F1_T0'] = e['F1']
        rows.append(row)
    fams = []
    for r in rows:
        if r['family'] not in fams:
            fams.append(r['family'])
    cols = VIEWS + [n for n in REF if any(f'{n} | ROC' in r for r in rows)]
    for metric, label in (('ROC', 'raw ROC-AUC'), ('F1_T0', 'ranked F1 at T = 0')):
        log(f'\n{label} (mean over seeds)')
        log(f"  {'family':<14}" + ''.join(f'{c:>22}' for c in cols))
        for fam in fams + ['ALL']:
            sel = rows if fam == 'ALL' else [r for r in rows if r['family'] == fam]
            log(f'  {fam:<14}' + ''.join(f'{np.mean([r.get(f"{c} | {metric}", np.nan) for r in sel]):>22.3f}' for c in cols))
    log('\nReliability weights (training data only): mean weight on the ANGLE register, and how many known families')
    log('the "best register" rule assigns to the angle register, per held-out family (mean over seeds)')
    log(f"  {'held-out family':<16}{'mean w_B':>10}{'families -> B':>16}")
    for fam in fams + ['ALL']:
        sel = rows if fam == 'ALL' else [r for r in rows if r['family'] == fam]
        log(f'  {fam:<16}{np.mean([r["mean weight on B (angle)"] for r in sel]):>10.3f}'
            f'{np.mean([r["families choosing B"] for r in sel]):>10.1f} / {rows[0]["known families"]}')
    dev = max(abs(r['equal (v3) | ROC'] - r['check: saved v3 ROC_raw']) for r in rows)
    log(f'\nConsistency check: max |re-scored equal-weight ROC - saved v3 ROC_raw| = {dev:.4f} (should be ~0)')
    log('\n"reliability-weighted" is the pre-specified candidate; "best register" is a sensitivity check only.')
    log('Weights use only known-family training samples; the held-out family never enters the weighting.')
    out = os.path.join(HERE, 'comparison'); os.makedirs(out, exist_ok=True)
    open(os.path.join(out, 'diagnosis_hybrid_v3.txt'), 'w').write('\n'.join(lines) + '\n')
    with open(os.path.join(out, 'diagnosis_hybrid_v3.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), restval=''); w.writeheader(); w.writerows(rows)
    print(f'\nSaved: {out}/diagnosis_hybrid_v3.txt and .csv')


if __name__ == '__main__':
    main()
