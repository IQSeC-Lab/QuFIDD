#!/usr/bin/env python3
"""
select_threshold.py -- capacity_matched (top level)   choose a quantum model's MAIN test thresholds on VALIDATION data
======================================================================================================================
For each of the three quantum threshold schemes, the threshold with the best mean FLAG-RULE F1 over the 12 validation
folds (pseudo-OOD, training data only) is selected:
  raw_rel7   RAW distance, observed range / 7   (k = 0 .. 6; k = 0 is T = 0; on test, k is applied to each task's range)
  mad_rel7   MAD score,    observed range / 7   (same)
  mad_fixed  MAD score, fixed T in {0, 1, 1.5, 2, 2.5, 3, 3.5}
Flag-rule F1 is the criterion because the ranked (CADE-protocol) F1 is maximal at T = 0 by construction.
Ties: the stricter threshold (higher k / T) is chosen. evaluate.py marks the selected rows with '*' on the test set.

Needs <model>/validation/scores_s*_f*.npz (written by the quantum scripts' 'validate' mode).
Usage (from capacity_matched/):
  python3 select_threshold.py pca8/quantum_angle_8
  python3 select_threshold.py pca64/quantum_amp_64
  python3 select_threshold.py pca64/quantum_amp_64 --steps 10          # other splits are possible
Output: <model>/validation/selected_thresholds.json (+ printed table)
"""
import os, glob, json, argparse
import numpy as np

MAD_FIXED = [0, 1, 1.5, 2, 2.5, 3, 3.5]


def f1s(score, raw, ood, T):
    flag = (score > T) if T > 0 else np.ones(len(score), bool); act = ood.astype(bool)
    tp, fp, fn = int((flag & act).sum()), int((flag & ~act).sum()), int((~flag & act).sum())
    P = tp / (tp + fp) if tp + fp else 0.0; R = tp / (tp + fn) if tp + fn else 0.0
    fl = 2 * P * R / (P + R) if P + R else 0.0
    idx = np.where(flag)[0]; rk = 0.0
    if len(idx):
        o = idx[np.argsort(-raw[idx], kind='stable')]; tpc = np.cumsum(ood[o]); k = np.arange(1, len(o) + 1)
        p, r = tpc / k, tpc / ood.sum(); rk = float(np.max(np.where(p + r > 0, 2 * p * r / (p + r + 1e-12), 0.0)))
    return fl, rk


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('model_dir'); ap.add_argument('--steps', type=int, default=7)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.model_dir, 'validation', 'scores_s*_f*.npz')))
    if not files:
        raise SystemExit(f'No validation scores in {a.model_dir}/validation -- run the validation array first.')
    N = a.steps; folds = [np.load(f) for f in files]
    schemes = {f'raw_rel{N}': ('raw', list(range(N))), f'mad_rel{N}': ('mad', list(range(N))), 'mad_fixed': ('mad', MAD_FIXED)}
    out = {'n_folds': len(files), 'steps': N, 'criterion': 'mean flag-rule F1 on validation (ties -> stricter)'}
    for key, (score, grid) in schemes.items():
        res = {}
        for g in grid:
            F, R, Tv = [], [], []
            for z in folds:
                raw, mad, ood = z['raw'], z['mad'], z['ood'].astype(int); sv = raw if score == 'raw' else mad
                T = float(g) if key == 'mad_fixed' else (0.0 if g == 0 else float(sv.min()) + g * (float(sv.max()) - float(sv.min())) / N)
                fl, rk = f1s(sv, raw, ood, T); F.append(fl); R.append(rk); Tv.append(T)
            res[g] = (np.mean(F), np.std(F), np.mean(R), np.mean(Tv))
        best = max(grid, key=lambda g: (round(res[g][0], 6), g))
        out[key] = best; out[key + '_val_flagF1'] = float(res[best][0]); out[key + '_val_rankedF1'] = float(res[best][2])
        print(f'\n{key}  ({score} score, {len(files)} validation folds)')
        print(f"  {'threshold':>10}{'mean T':>9}{'flag-rule F1':>18}{'ranked F1':>12}")
        for g in grid:
            lab = f'T={g:g}' if key == 'mad_fixed' else f'k={g}'
            print(f'  {lab:>10}{res[g][3]:>9.3f}{res[g][0]:>11.3f}+-{res[g][1]:<5.3f}{res[g][2]:>10.3f}' + ('   <- selected' if g == best else ''))
    p = os.path.join(a.model_dir, 'validation', 'selected_thresholds.json'); json.dump(out, open(p, 'w'), indent=2)
    print(f'\nSaved: {p}')


if __name__ == '__main__':
    main()
