#!/usr/bin/env python3
"""
validation_compare.py -- pca64_experiment (top level)
=====================================================
Side-by-side VALIDATION results (training data only; test never used) for the quantum model, CADE-mini 64-3-3
and the PCA-64 L2 baseline on the same 12 folds, at ALL thresholds:
  ROC-AUC (raw, MAD), PR-AUC; CADE-protocol F1 at T in {0, 2, 2.5, 3, 3.5}; fixed-threshold F1 at T = 2 and 3.5;
  CALIBRATED thresholds per model (95th percentile of its own TRAINING scores = the same 5% false-alarm budget
  for every model), on the MAD score (T_cal) and on the RAW distance (T_rawcal): F1 (CADE protocol and fixed),
  precision, recall and the realized false-alarm rate.
Per metric: mean +- std over folds, and per-fold wins quantum vs classical.
Usage (from pca64_experiment/, after both validation arrays finished):  python3 validation_compare.py
"""
import os, json, glob
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
KEYS = [('ROC_raw', 'ROC-AUC (raw)'), ('ROC_mad', 'ROC-AUC (MAD score)'), ('PR_raw', 'PR-AUC (raw)'),
        ('F1@T0.0', 'F1 threshold-free (T=0)'), ('F1@T2.0', 'F1 CADE protocol T=2.0'), ('F1@T2.5', 'F1 CADE protocol T=2.5'),
        ('F1@T3.0', 'F1 CADE protocol T=3.0'), ('F1@T3.5', 'F1 CADE protocol T=3.5'), ('fixedF1@T2.0', 'F1 fixed T=2.0'),
        ('fixedF1@T3.5', 'F1 fixed T=3.5'), ('T_cal', 'calibrated T_cal'), ('F1@Tcal', 'F1 CADE protocol T=T_cal'),
        ('fixedF1@Tcal', 'F1 fixed T=T_cal'), ('FPR@Tcal', 'false-alarm rate at T_cal'),
        ('T_rawcal', 'raw-distance T_rawcal'), ('F1@RawCal', 'F1 CADE protocol raw-cal'), ('fixedF1@RawCal', 'F1 fixed raw-cal'),
        ('P@RawCal', 'precision raw-cal'), ('R@RawCal', 'recall raw-cal'), ('FPR@RawCal', 'false-alarm rate raw-cal')]
load = lambda d: {(r['split'], r['fold']): r for r in (json.load(open(f)) for f in glob.glob(os.path.join(HERE, d, 'validation', 's*_f*.json')))}
Q, C = load('quantum_amp_64'), load('cade_mini_64')
common = sorted(set(Q) & set(C))
print('=' * 112)
print(f'PCA-64 VALIDATION (training data only; test never used): {len(common)} shared folds')
print('=' * 112)
if common and 'F1@RawCal' not in Q[common[0]]['model']:
    raise SystemExit('Validation files are from an older version (no raw-distance calibration). Rerun both validation arrays.')
print(f"  {'metric':<28}{'PCA-64 L2 (0)':>18}{'CADE-mini (475)':>18}{'Quantum (474)':>18}{'Q wins':>9}")
for k, lab in KEYS:
    b = [Q[x]['pca64_l2_baseline'][k] for x in common]; c = [C[x]['model'][k] for x in common]; q = [Q[x]['model'][k] for x in common]
    wins = '' if k in ('T_cal', 'FPR@Tcal', 'T_rawcal', 'FPR@RawCal') else f'{sum(a > d + 1e-12 for a, d in zip(q, c))}/{len(common)}'
    print(f'  {lab:<28}' + ''.join(f'{np.mean(v):>11.3f}+-{np.std(v):<5.3f}' for v in (b, c, q)) + f'{wins:>9}')
print('\n  "Q wins" = folds where the quantum model beats CADE-mini. Validation guides design only; the test set')
print('  is evaluated once, after the design is fixed.')
