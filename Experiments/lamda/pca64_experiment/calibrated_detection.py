#!/usr/bin/env python3
"""
calibrated_detection.py -- detection at CALIBRATED thresholds, for every model in an experiment folder
======================================================================================================
A threshold is calibrated on KNOWN TRAINING data only: T = the (1 - target) quantile of the model's own training
scores, so every model flags the same share (target) of its known training samples. The test set and the held-out
family are never used to set T. Reported for each target false-alarm rate:
  T          the calibrated threshold (mean over tasks)
  FPR        realized false-alarm rate = share of KNOWN TEST samples flagged (should be close to the target)
  P, R, F1   precision / recall / F1 of the flagged set (fixed threshold, flag only)
  CADE-F1    CADE protocol inside the flagged set (flagged samples ranked by raw distance, best cut-off)

Which scores are calibrated:
  quantum models (folder name contains 'quantum')  -> BOTH the MAD score and the RAW distance
                                                       (bounded distances cap the MAD score; see the paper text)
  classical models and zero-parameter baselines    -> the MAD score (CADE's detection score)

Training scores come from the files the models saved (train_anomaly_scores.npy, train_raw_scores.npy); if missing,
they are recomputed from saved states (quantum), saved weights (CADE-minis) or the data (PCA baselines).

Usage (from lamda/):
  python3 calibrated_detection.py pca64_experiment
  python3 calibrated_detection.py Full_new_experiments --targets 0.01 0.05 0.10 --by-family
  python3 calibrated_detection.py pca8_experiment --data-dir ~/projects/OOD/Data/cade_lamda_4class_vt
Outputs: <experiment>/calibrated_detection.txt and .csv
"""
import os, re, glob, json, csv, argparse
import numpy as np

MAD_B = 1.4826


# ----------------------------------------------------------------------------- scoring helpers
def mad_stats(d):
    med = float(np.median(d)); return med, max(MAD_B * float(np.median(np.abs(d - med))), 1e-12)


def mad_A(own, D):
    A = []
    for c, d in enumerate(own):
        med, mad = mad_stats(d); A.append(np.abs(D[:, c] - med) / mad)
    return np.min(A, axis=0)


def quantum_train_scores(S, y):
    """Mixed density-matrix prototypes, leave-one-out for the own class: returns (MAD scores, raw distances)."""
    C = len(np.unique(y)); D = np.zeros((len(y), C)); own = []
    for c in range(C):
        m = y == c; F = np.abs(S.conj() @ S[m].T) ** 2; n = m.sum()
        Dc = 1.0 - F.mean(axis=1); Dc[m] = 1.0 - (F[m].sum(axis=1) - 1.0) / (n - 1)
        D[:, c] = Dc; own.append(Dc[m])
    return mad_A(own, D), D.min(axis=1)


def centroid_train_scores(L, y):
    C = len(np.unique(y)); cents = [L[y == c].mean(0) for c in range(C)]
    D = np.stack([np.linalg.norm(L - mu, axis=1) for mu in cents], axis=1)
    return mad_A([D[y == c, c] for c in range(C)], D), D.min(axis=1)


# ----------------------------------------------------------------------------- data (only when recomputation is needed)
def load_split(data_dir, prefix, s):
    d = np.load(os.path.join(os.path.expanduser(data_dir), f'{prefix}_{s}.npz'), allow_pickle=True)
    dense = lambda a: a.item().toarray().astype(np.float32) if a.shape == () else a.astype(np.float32)
    ytr = d['y_train'].astype(np.int64); known = sorted(np.unique(ytr).tolist())
    return dense(d['X_train']), np.array([known.index(int(c)) for c in ytr])


def encoder_from_weights(path):
    """Rebuild a CADE-mini encoder from its saved state dict (architecture read from the weight shapes)."""
    import torch, torch.nn as nn
    sd = torch.load(path, map_location='cpu')
    ks = sorted({int(k.split('.')[1]) for k in sd if k.startswith('0.') and k.endswith('.weight')})
    layers = []
    for i, k in enumerate(ks):
        W = sd[f'0.{k}.weight']; lin = nn.Linear(W.shape[1], W.shape[0])
        lin.weight.data, lin.bias.data = W.clone(), sd[f'0.{k}.bias'].clone(); layers.append(lin)
        if i < len(ks) - 1:
            layers.append(nn.ReLU())
    enc = nn.Sequential(*layers)
    return lambda Z: enc(torch.from_numpy(np.asarray(Z, np.float32))).detach().numpy(), int(sd[f'0.{ks[0]}.weight'].shape[1])


# ----------------------------------------------------------------------------- metrics
def detect(score, raw, ood, T):
    flag = score > T; act = ood.astype(bool)
    tp, fp, fn = int((flag & act).sum()), int((flag & ~act).sum()), int((~flag & act).sum())
    P = tp / (tp + fp) if tp + fp else 0.0; R = tp / (tp + fn) if tp + fn else 0.0
    idx = np.where(flag)[0]; cade = 0.0
    if len(idx):
        o = idx[np.argsort(-raw[idx], kind='stable')]; tpc = np.cumsum(ood[o]); k = np.arange(1, len(o) + 1)
        p, r = tpc / k, tpc / ood.sum(); cade = float(np.max(np.where(p + r > 0, 2 * p * r / (p + r + 1e-12), 0.0)))
    return {'T': T, 'FPR': float(flag[~act].mean()), 'P': P, 'R': R, 'F1': 2 * P * R / (P + R) if P + R else 0.0, 'CADE-F1': cade}


# ----------------------------------------------------------------------------- discovery
def pca_dim_of(exp):
    m = re.search(r'pca(\d+)', os.path.basename(exp).lower())
    return int(m.group(1)) if m else 5


def discover(exp):
    out = {}
    for d in sorted(glob.glob(os.path.join(exp, '*', 'results*'))):
        files = sorted(glob.glob(os.path.join(d, 'run*', 'split*', 'ood_scores.csv')))
        if not files:
            continue
        folder, res = os.path.relpath(d, exp).split(os.sep)
        suffix = res[len('results'):].lstrip('_')
        label = folder + ('' if not suffix else (' (L2)' if suffix == 'l2' else f' ({suffix})'))
        out[label] = {'folder': folder, 'suffix': suffix, 'quantum': 'quantum' in folder.lower(),
                      'baseline': folder.startswith('pca'), 'tasks': files}
    return out


def train_scores(model, tdir, run, split, args, cache):
    """-> (MAD training scores, raw training distances or None)."""
    fA, fR = os.path.join(tdir, 'train_anomaly_scores.npy'), os.path.join(tdir, 'train_raw_scores.npy')
    A = np.load(fA) if os.path.exists(fA) else None
    R = np.load(fR) if os.path.exists(fR) else None
    if model['quantum'] and (A is None or R is None):
        for fn, key in [('features.npz', 'features_train'), ('states.npz', 'states_train')]:
            p = os.path.join(tdir, fn)
            if os.path.exists(p):
                z = np.load(p)
                if np.iscomplexobj(z[key]):
                    A2, R2 = quantum_train_scores(z[key], z['y_train'].astype(int))
                    A = A if A is not None else A2; R = R if R is not None else R2
                break
    if A is None and not model['quantum']:
        key = (split,)
        if key not in cache:
            from sklearn.decomposition import PCA
            X, y = load_split(args.data_dir, args.prefix, split)
            cache[key] = (X, y, PCA(n_components=args.pca_dim, svd_solver='full').fit(X))
        X, y, pca = cache[key]; Z = pca.transform(X)
        if model['baseline']:
            if model['suffix'] == 'l2':
                Z = Z / np.linalg.norm(Z, axis=1, keepdims=True).clip(min=1e-8)
            A, R2 = centroid_train_scores(Z, y)
        else:
            w = os.path.join(tdir, 'model_weights.pt')
            if not os.path.exists(w):
                return None, None
            f, _ = encoder_from_weights(w); A, R2 = centroid_train_scores(f(Z), y)
    return A, R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('experiment')
    ap.add_argument('--targets', type=float, nargs='+', default=[0.01, 0.05, 0.10])
    ap.add_argument('--by-family', action='store_true')
    ap.add_argument('--data-dir', default='~/projects/OOD/Data/cade_lamda_4class_vt')
    ap.add_argument('--prefix', default='drebin_new')
    ap.add_argument('--pca-dim', type=int, default=None)
    args = ap.parse_args()
    exp = os.path.abspath(args.experiment); args.pca_dim = args.pca_dim or pca_dim_of(exp)
    models = discover(exp)
    if not models:
        raise SystemExit(f'No results found under {exp}')
    lines, rows, cache = [], [], {}
    def log(s=''):
        print(s, flush=True); lines.append(s)
    cols = ['T', 'FPR', 'P', 'R', 'F1', 'CADE-F1']
    log('=' * 118)
    log(f'CALIBRATED DETECTION -- {os.path.basename(exp)} (T = (1 - target) quantile of each model\'s KNOWN TRAINING scores; test scores)')
    log('Quantum models: MAD score AND raw distance. Classical models and baselines: MAD score.  mean +- std over tasks')
    log('=' * 118)
    for label, model in models.items():
        kinds = ['MAD', 'raw'] if model['quantum'] else ['MAD']
        per = {(k, t): [] for k in kinds for t in args.targets}; fam_of = {}
        for f in model['tasks']:
            tdir = os.path.dirname(f)
            run = int(re.search(r'run(\d+)', f).group(1)); split = int(re.search(r'split(\d+)', f).group(1))
            a = np.genfromtxt(f, delimiter=',', names=True)
            raw, mad, ood = a['raw_score'], a['mad_score'], a['is_ood'].astype(int)
            sj = os.path.join(tdir, 'summary.json'); fam = json.load(open(sj)).get('ood_family', '?') if os.path.exists(sj) else '?'
            A, R = train_scores(model, tdir, run, split, args, cache)
            for k in kinds:
                tr, te = (A, mad) if k == 'MAD' else (R, raw)
                if tr is None:
                    continue
                for t in args.targets:
                    r = detect(te, raw, ood, float(np.quantile(tr, 1 - t))); r['family'] = fam; per[(k, t)].append(r)
        log(f'\n  {label}  ({len(model["tasks"])} tasks){"  [quantum: MAD + raw]" if model["quantum"] else ""}')
        log(f"    {'score':<6}{'target':>8}" + ''.join(f'{c:>16}' for c in cols))
        for k in kinds:
            for t in args.targets:
                res = per[(k, t)]
                if not res:
                    log(f'    {k:<6}{t:>8.2f}   (no training scores found -- rerun the model or pass --data-dir)'); continue
                M = {c: [x[c] for x in res] for c in cols}
                log(f'    {k:<6}{t:>8.2f}' + ''.join(f'{np.mean(M[c]):>9.3f}+-{np.std(M[c]):<5.3f}' for c in cols))
                rows.append({'model': label, 'score': k, 'target': t, 'family': 'ALL', 'tasks': len(res), **{c: np.mean(M[c]) for c in cols}})
                if args.by_family:
                    for fam in sorted({x['family'] for x in res}):
                        rf = [x for x in res if x['family'] == fam]; Mf = {c: [x[c] for x in rf] for c in cols}
                        log(f'      {fam:<12}' + ''.join(f'{np.mean(Mf[c]):>9.3f}+-{np.std(Mf[c]):<5.3f}' for c in cols))
                        rows.append({'model': label, 'score': k, 'target': t, 'family': fam, 'tasks': len(rf), **{c: np.mean(Mf[c]) for c in cols}})
    log('\n  target = share of known TRAINING samples flagged by construction; FPR = realized share of known TEST samples')
    log('  flagged (close to the target = the calibration transfers); P/R/F1 = flagged set; CADE-F1 = CADE protocol at T.')
    open(os.path.join(exp, 'calibrated_detection.txt'), 'w').write('\n'.join(lines) + '\n')
    with open(os.path.join(exp, 'calibrated_detection.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['model', 'score', 'target', 'family', 'tasks'] + cols); w.writeheader(); w.writerows(rows)
    print(f'\nSaved: {exp}/calibrated_detection.txt, .csv')


if __name__ == '__main__':
    main()
