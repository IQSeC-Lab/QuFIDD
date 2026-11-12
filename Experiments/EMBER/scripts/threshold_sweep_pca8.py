#!/usr/bin/env python3
"""
threshold_sweep_pca8.py -- Experiments2/EMBER/pca8: EMBER 2018 (8 families, capped 1000, CADE per-split cut)   (standalone, no retraining)
====================================================================================================================
Threshold sweeps on the saved scores of every PCA-8 model, for two scores:

  RAW  : thresholds = q-th percentile of the model's own TRAINING raw scores (nearest-family distance),
         q in {0, 50, 75, 90, 95, 97.5, 99}; q = 0 flags every test sample. Nominal FPR on known families = 1 - q/100.
  MAD  : CADE's score A = min_c |d_c - median_c| / MAD_c (median/MAD of each family's training distances to its own
         centroid / prototype, MAD constant 1.4826), thresholds T in {0, 1, 1.5, 2, 2.5, 3, 3.5}; T = 0 flags all.

No test data is used to set any threshold. At each threshold the flagged test samples are evaluated two ways:
  ranked   : flagged samples ranked by raw distance, F1 at the best cut-off (CADE's evaluation)
  flagged  : every flagged sample predicted as drift (F1 over all flagged samples)
plus the fraction of test samples flagged, the observed FPR vs. the nominal / training flag rate.

Score sources (per task results/run<r>/split<s>/), detected automatically:
  PCA-only baseline        : no weights -> PCA-8 refit on the split's training data, train/test scores recomputed
                             (variant raw/L2 identified by reproducing the saved test raw scores)
  classical models + twins : ood_scores.csv (raw_score, mad_score, is_ood) + train_raw_scores.npy + train_anomaly_scores.npy
  PennyLane quantum models : states.npz (states_train_r*, states_test_r*, y_train) + ood_scores.csv (is_ood);
                             pure-state prototypes, raw and MAD scores recomputed exactly as in the twins
                             (the recomputed test raw score is checked against ood_scores.csv)

Outputs, for every model folder <model>/results/:
  Threshold_sweep_raw/{per_task.csv, aggregate.csv, summary.txt, plots/*.png}
  Threshold_sweep_MAD/{per_task.csv, aggregate.csv, summary.txt, plots/*.png}
and an all-model comparison in pca8/threshold_sweep_summary/{raw_all_models.csv, MAD_all_models.csv, summary.txt, plots}.

Usage (from EMBER/pca8/):
  python3 threshold_sweep_pca8.py                       all model folders under quantum_pca8, cade_pca8_*, twin_pca8_*
  python3 threshold_sweep_pca8.py --include-ablation    also ablation_pca8
  python3 threshold_sweep_pca8.py --models twin_pca8_p702/classical_fidtwin_c_702 ...   only these model folders
"""
import os, sys, csv, glob, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
Q_RAW = [0.0, 50.0, 75.0, 90.0, 95.0, 97.5, 99.0]
T_MAD = [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
MAD_B = 1.4826
DEFAULT_GROUPS = ['quantum_pca8', 'cade_pca8_*', 'twin_pca8_*']
METRICS = ['ranked_F1', 'ranked_Precision', 'ranked_Recall', 'ranked_FPR', 'ranked_FNR', 'ranked_Effort',
           'flagged_F1', 'flagged_Precision', 'flagged_Recall', 'flagged_FPR', 'flagged_FNR',
           'Frac_flagged', 'train_flag_rate']


# ----------------------------------------------------------------------------- scores
def _mad_stats(d):
    med = float(np.median(d)); mad = float(MAD_B * np.median(np.abs(d - med)))
    return med, (mad if mad > 0 else 1e-12)


def _pure_prototypes(S, y, C):
    V = []
    for c in range(C):
        P = S[y == c]; rho = P.T @ P.conj() / len(P)
        V.append(np.linalg.eigh(rho)[1][:, -1].conj())
    return np.stack(V)


def _proto_scores(Str_list, ytr, Ste_list):
    """d_c = 1 - prod_r |<v_rc|psi_r>|^2 (pure-state prototypes); same rule as the twins' proto_detect."""
    C = len(np.unique(ytr)); Ptr, Pte = 1.0, 1.0
    for S, Se in zip(Str_list, Ste_list):
        V = _pure_prototypes(S, ytr, C)
        Ptr = Ptr * np.abs(S @ V.T) ** 2; Pte = Pte * np.abs(Se @ V.T) ** 2
    Dtr, Dte = 1.0 - Ptr, 1.0 - Pte; Atr, Ate = [], []
    for c in range(C):
        med, mad = _mad_stats(Dtr[ytr == c, c])
        Atr.append(np.abs(Dtr[:, c] - med) / mad); Ate.append(np.abs(Dte[:, c] - med) / mad)
    return Dtr.min(1), np.min(Atr, axis=0), Dte.min(1), np.min(Ate, axis=0)


def _read_scores_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    cols = rows[0].keys()
    get = lambda k, t=float: np.array([t(r[k]) for r in rows]) if k in cols else None
    return get('raw_score'), get('mad_score'), get('is_ood', lambda v: int(float(v)))


def load_task(task_dir):
    """Return dict(raw_te, mad_te, raw_tr, mad_tr, is_ood, source) or (None, reason)."""
    sc = os.path.join(task_dir, 'ood_scores.csv')
    if not os.path.exists(sc):
        return None, 'no ood_scores.csv'
    raw_te, mad_te, is_ood = _read_scores_csv(sc)
    if is_ood is None:
        return None, 'ood_scores.csv has no is_ood column'
    f_rtr, f_atr = os.path.join(task_dir, 'train_raw_scores.npy'), os.path.join(task_dir, 'train_anomaly_scores.npy')
    if mad_te is not None and os.path.exists(f_rtr) and os.path.exists(f_atr):
        return {'raw_te': raw_te, 'mad_te': mad_te, 'raw_tr': np.load(f_rtr), 'mad_tr': np.load(f_atr),
                'is_ood': is_ood, 'source': 'saved scores'}, None
    f_st = os.path.join(task_dir, 'states.npz')
    if os.path.exists(f_st):
        z = np.load(f_st, allow_pickle=True)
        regs = sorted(int(k.split('_r')[-1]) for k in z.files if k.startswith('states_train_r'))
        if regs and 'y_train' in z.files:
            Str = [z[f'states_train_r{r}'] for r in regs]; Ste = [z[f'states_test_r{r}'] for r in regs]
            rtr, atr, rte, ate = _proto_scores(Str, z['y_train'].astype(int), Ste)
            if raw_te is not None and len(raw_te) == len(rte):
                diff = float(np.max(np.abs(raw_te - rte)))
                if diff > 1e-4:
                    print(f'    [warn] recomputed raw score differs from ood_scores.csv by {diff:.2e} in {task_dir}')
            return {'raw_te': rte, 'mad_te': ate, 'raw_tr': rtr, 'mad_tr': atr, 'is_ood': is_ood,
                    'source': 'recomputed from states.npz'}, None
    if 'pca_only' in task_dir:
        return _pca_only_task(task_dir, raw_te, is_ood)
    return None, 'no train scores and no states.npz'


# ----------------------------------------------------------------------------- PCA-only baseline (no weights)
DATA_DIR, DATA_PREFIX, PCA_DIM = '/home/snowm001/projects/OOD/Data2/EMBER/cade_ember2018_cadesplit_1000_k8', 'drebin_new', 8
_PCA_CACHE = {}


def _centroid_scores(Ztr, ytr, Zte):
    """CADE detect: Euclidean distance to the nearest known-family centroid + per-family MAD score."""
    D_tr, D_te, A_tr, A_te = [], [], [], []
    for c in range(len(np.unique(ytr))):
        mu = Ztr[ytr == c].mean(0); med, mad = _mad_stats(np.linalg.norm(Ztr[ytr == c] - mu, axis=1))
        dtr, dte = np.linalg.norm(Ztr - mu, axis=1), np.linalg.norm(Zte - mu, axis=1)
        D_tr.append(dtr); D_te.append(dte); A_tr.append(np.abs(dtr - med) / mad); A_te.append(np.abs(dte - med) / mad)
    return np.min(D_tr, 0), np.min(A_tr, 0), np.min(D_te, 0), np.min(A_te, 0)


def _pca_only_task(task_dir, raw_saved, is_ood):
    """No trained weights: refit PCA-8 on the split's training data (deterministic, seed-independent) and recompute
    train/test raw and MAD scores. The variant (raw coordinates or L2-normalized) is the one that reproduces the
    saved test raw scores."""
    from sklearn.decomposition import PCA
    split = int(os.path.basename(task_dir.rstrip('/')).replace('split', ''))
    if split not in _PCA_CACHE:
        d = np.load(os.path.join(DATA_DIR, f'{DATA_PREFIX}_{split}.npz'), allow_pickle=True)
        dense = lambda a: a.item().toarray().astype(np.float32) if a.shape == () else a.astype(np.float32)
        Xtr, Xte, ytr = dense(d['X_train']), dense(d['X_test']), d['y_train'].astype(np.int64)
        remap = {c: i for i, c in enumerate(sorted(np.unique(ytr)))}; ytr = np.array([remap[c] for c in ytr])
        pca = PCA(n_components=PCA_DIM, svd_solver='full').fit(Xtr); Ztr, Zte = pca.transform(Xtr), pca.transform(Xte)
        l2 = lambda Z: Z / np.linalg.norm(Z, axis=1, keepdims=True).clip(min=1e-8)
        _PCA_CACHE[split] = {'coordinates': _centroid_scores(Ztr, ytr, Zte), 'L2': _centroid_scores(l2(Ztr), ytr, l2(Zte))}
    variants = _PCA_CACHE[split]
    if raw_saved is None:
        return None, 'pca_only: ood_scores.csv has no raw_score to identify the variant'
    diffs = {k: (float(np.max(np.abs(v[2] - raw_saved))) if len(v[2]) == len(raw_saved) else np.inf) for k, v in variants.items()}
    best = min(diffs, key=diffs.get)
    if diffs[best] > 1e-3:
        return None, f'pca_only: recomputed scores do not match saved ones (max diff {diffs})'
    rtr, atr, rte, ate = variants[best]
    return {'raw_te': rte, 'mad_te': ate, 'raw_tr': rtr, 'mad_tr': atr, 'is_ood': is_ood,
            'source': f'recomputed PCA-only ({best}, max diff {diffs[best]:.1e})'}, None


# ----------------------------------------------------------------------------- metrics
def _ranked(raw, is_ood, idx):
    n_ood, n_known = int(is_ood.sum()), int(len(is_ood) - is_ood.sum())
    if len(idx) == 0:
        return dict(F1=0.0, Precision=0.0, Recall=0.0, FPR=0.0, FNR=1.0, Effort=0.0)
    order = idx[np.argsort(-raw[idx], kind='stable')]
    tp = np.cumsum(is_ood[order]); k = np.arange(1, len(order) + 1); p, r = tp / k, tp / n_ood
    f = np.where(p + r > 0, 2 * p * r / (p + r + 1e-12), 0.0); b = int(np.argmax(f))
    return dict(F1=float(f[b]), Precision=float(p[b]), Recall=float(r[b]), FPR=float((k[b] - tp[b]) / n_known),
                FNR=float(1 - r[b]), Effort=float(k[b] / n_ood))


def _flagged(is_ood, flag):
    act = is_ood.astype(bool); tp = int((flag & act).sum()); fp = int((flag & ~act).sum()); fn = int((~flag & act).sum())
    P = tp / (tp + fp) if tp + fp else 0.0; R = tp / (tp + fn) if tp + fn else 0.0
    return dict(F1=(2 * P * R / (P + R) if P + R else 0.0), Precision=P, Recall=R,
                FPR=fp / max(1, int((~act).sum())), FNR=1 - R)


def sweep_task(t):
    raw_rows, mad_rows = [], []
    for q in Q_RAW:
        tau = -np.inf if q == 0 else float(np.percentile(t['raw_tr'], q))
        flag = np.ones(len(t['raw_te']), bool) if q == 0 else t['raw_te'] > tau
        rk, fl = _ranked(t['raw_te'], t['is_ood'], np.where(flag)[0]), _flagged(t['is_ood'], flag)
        raw_rows.append({'q': q, 'threshold': (float('nan') if q == 0 else tau), 'nominal_FPR': 1 - q / 100,
                         **{f'ranked_{k}': v for k, v in rk.items()}, **{f'flagged_{k}': v for k, v in fl.items()},
                         'Frac_flagged': float(flag.mean()),
                         'train_flag_rate': 1.0 if q == 0 else float((t['raw_tr'] > tau).mean())})
    for T in T_MAD:
        flag = np.ones(len(t['mad_te']), bool) if T == 0 else t['mad_te'] > T
        rk, fl = _ranked(t['raw_te'], t['is_ood'], np.where(flag)[0]), _flagged(t['is_ood'], flag)
        mad_rows.append({'T': T, **{f'ranked_{k}': v for k, v in rk.items()}, **{f'flagged_{k}': v for k, v in fl.items()},
                         'Frac_flagged': float(flag.mean()),
                         'train_flag_rate': 1.0 if T == 0 else float((t['mad_tr'] > T).mean())})
    return raw_rows, mad_rows


# ----------------------------------------------------------------------------- per-model output
def _write_csv(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def _aggregate(per_task, key, levels):
    agg = []
    for lv in levels:
        R = [r for r in per_task if r[key] == lv]
        row = {key: lv, 'n_tasks': len(R)}
        if key == 'q':
            row['nominal_FPR'] = 1 - lv / 100
        for m in METRICS:
            v = np.array([r[m] for r in R], float)
            row[f'{m}_mean'], row[f'{m}_std'] = float(v.mean()), float(v.std())
        agg.append(row)
    return agg


def _plots(out, agg, key, label, name):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    os.makedirs(os.path.join(out, 'plots'), exist_ok=True)
    x = np.arange(len(agg)); xt = [('all' if a[key] == 0 else f'{a[key]:g}') for a in agg]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for m, st in [('ranked_F1', '-o'), ('flagged_F1', '--s'), ('flagged_Recall', ':^'), ('Frac_flagged', '-.v')]:
        mu = np.array([a[f'{m}_mean'] for a in agg]); sd = np.array([a[f'{m}_std'] for a in agg])
        ax.plot(x, mu, st, label=m); ax.fill_between(x, mu - sd, mu + sd, alpha=0.12)
    ax.set_xticks(x); ax.set_xticklabels(xt); ax.set_xlabel(label); ax.set_ylim(0, 1.02); ax.grid(alpha=0.3)
    ax.set_title(name, fontsize=9); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(out, 'plots', 'f1_vs_threshold.png'), dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ref = [a['nominal_FPR'] for a in agg] if key == 'q' else [a['train_flag_rate_mean'] for a in agg]
    ax.plot(x, ref, '--k', label='nominal (1 - q/100)' if key == 'q' else 'training flag rate')
    mu = np.array([a['flagged_FPR_mean'] for a in agg]); sd = np.array([a['flagged_FPR_std'] for a in agg])
    ax.plot(x, mu, '-o', color='tab:red', label='observed test FPR'); ax.fill_between(x, mu - sd, mu + sd, color='tab:red', alpha=0.12)
    ax.set_xticks(x); ax.set_xticklabels(xt); ax.set_xlabel(label); ax.set_ylabel('false positive rate')
    ax.set_ylim(0, 1.02); ax.grid(alpha=0.3); ax.set_title(name, fontsize=9); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(out, 'plots', 'fpr_observed_vs_nominal.png'), dpi=150); plt.close(fig)


def _summary_txt(path, name, agg, key, label, n_found, n_skip, sources):
    L = [f'{name}', f'{label} sweep  ({n_found} tasks, {n_skip} skipped; scores: {", ".join(sorted(sources))})',
         'mean +- std over tasks; ranked = flagged samples ranked by raw distance, best cut-off; flagged = all flagged = drift', '']
    hdr = f'{key:>6} ' + (f'{"nomFPR":>7} ' if key == 'q' else f'{"trFlag":>7} ') + \
          f'{"ranked_F1":>15} {"flagged_F1":>15} {"flag_Prec":>15} {"flag_Rec":>15} {"obs_FPR":>15} {"Frac_flag":>15}'
    L.append(hdr); L.append('-' * len(hdr))
    pm = lambda a, m: f'{a[m + "_mean"]:.3f}+-{a[m + "_std"]:.3f}'
    for a in agg:
        ref = a['nominal_FPR'] if key == 'q' else a['train_flag_rate_mean']
        L.append(f'{("all" if a[key] == 0 else f"{a[key]:g}"):>6} {ref:7.3f} ' + ' '.join(f'{pm(a, m):>15}' for m in
                 ['ranked_F1', 'flagged_F1', 'flagged_Precision', 'flagged_Recall', 'flagged_FPR', 'Frac_flagged']))
    txt = '\n'.join(L) + '\n'; open(path, 'w').write(txt); return txt


def model_name(model_dir):
    for f in sorted(glob.glob(os.path.join(model_dir, 'results', 'run*', 'split*', 'summary.json')))[:1]:
        try:
            return json.load(open(f)).get('model', os.path.relpath(model_dir, HERE))
        except Exception:
            pass
    return os.path.relpath(model_dir, HERE)


def run_model(model_dir):
    rel = os.path.relpath(model_dir, HERE); name = model_name(model_dir)
    tasks = sorted(glob.glob(os.path.join(model_dir, 'results', 'run*', 'split*')))
    raw_all, mad_all, sources, skipped = [], [], set(), []
    for td in tasks:
        t, why = load_task(td)
        if t is None:
            skipped.append((os.path.relpath(td, model_dir), why)); continue
        run_s, split_s = td.rstrip('/').split(os.sep)[-2:]
        meta = {'run': run_s, 'split': split_s}
        try:
            meta['ood_family'] = json.load(open(os.path.join(td, 'summary.json'))).get('ood_family', '')
        except Exception:
            meta['ood_family'] = ''
        rr, mr = sweep_task(t); sources.add(t['source'])
        raw_all += [{**meta, **r} for r in rr]; mad_all += [{**meta, **r} for r in mr]
    print(f'\n== {rel}  ({name})\n   {len(tasks) - len(skipped)}/{len(tasks)} tasks usable' +
          (f'; skipped e.g. {skipped[0][0]}: {skipped[0][1]}' if skipped else ''))
    if not raw_all:
        return None
    n_ok = len(tasks) - len(skipped); out = {}
    for tag, rows, key, levels, label in [('raw', raw_all, 'q', Q_RAW, 'percentile q of training raw scores'),
                                          ('MAD', mad_all, 'T', T_MAD, 'MAD threshold T')]:
        d = os.path.join(model_dir, 'results', f'Threshold_sweep_{tag}'); os.makedirs(d, exist_ok=True)
        _write_csv(os.path.join(d, 'per_task.csv'), rows)
        agg = _aggregate(rows, key, levels); _write_csv(os.path.join(d, 'aggregate.csv'), agg)
        print(_summary_txt(os.path.join(d, 'summary.txt'), name, agg, key, label, n_ok, len(skipped), sources))
        _plots(d, agg, key, label, name); out[tag] = agg
        print(f'   saved -> {os.path.relpath(d, HERE)}/')
    return {'model': name, 'folder': rel, 'n_tasks': n_ok, **out}


# ----------------------------------------------------------------------------- all-model comparison
def write_comparison(results):
    d = os.path.join(HERE, 'threshold_sweep_summary'); os.makedirs(os.path.join(d, 'plots'), exist_ok=True)
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    txt = []
    for tag, key, levels, label in [('raw', 'q', Q_RAW, 'percentile q of training raw scores'), ('MAD', 'T', T_MAD, 'MAD threshold T')]:
        rows = [{'model': r['model'], 'folder': r['folder'], 'n_tasks': r['n_tasks'], **a} for r in results for a in r[tag]]
        _write_csv(os.path.join(d, f'{tag}_all_models.csv'), rows)
        txt.append(f'== {tag} sweep: ranked F1 (mean over tasks) | observed FPR in brackets')
        txt.append(f'{"model":<60} ' + ' '.join(f'{("all" if lv == 0 else f"{key}={lv:g}"):>14}' for lv in levels))
        for r in results:
            txt.append(f'{r["folder"]:<60} ' + ' '.join(f'{a["ranked_F1_mean"]:.3f} ({a["flagged_FPR_mean"]:.2f})'.rjust(14) for a in r[tag]))
        txt.append('')
        for m, fn in [('ranked_F1', 'ranked_F1'), ('flagged_F1', 'flagged_F1'), ('flagged_FPR', 'observed_FPR')]:
            fig, ax = plt.subplots(figsize=(8, 4.5)); x = np.arange(len(levels))
            for r in results:
                ax.plot(x, [a[f'{m}_mean'] for a in r[tag]], '-o', ms=3, label=r['folder'].split('/')[-1])
            if m == 'flagged_FPR' and tag == 'raw':
                ax.plot(x, [1 - q / 100 for q in levels], '--k', label='nominal')
            ax.set_xticks(x); ax.set_xticklabels([('all' if lv == 0 else f'{lv:g}') for lv in levels])
            ax.set_xlabel(label); ax.set_ylabel(m); ax.set_ylim(0, 1.02); ax.grid(alpha=0.3)
            ax.legend(fontsize=6, ncol=2); fig.tight_layout()
            fig.savefig(os.path.join(d, 'plots', f'{tag}_{fn}.png'), dpi=150); plt.close(fig)
    open(os.path.join(d, 'summary.txt'), 'w').write('\n'.join(txt) + '\n')
    print('\n' + '\n'.join(txt) + f'\nSaved: {os.path.relpath(d, HERE)}/{{raw_all_models.csv, MAD_all_models.csv, summary.txt, plots/}}')


def find_models(groups):
    found = []
    for g in groups:
        for base in sorted(glob.glob(os.path.join(HERE, g))):
            for res in sorted(glob.glob(os.path.join(base, '**', 'results'), recursive=True)):
                if 'Threshold_sweep' in res:
                    continue
                if glob.glob(os.path.join(res, 'run*', 'split*')):
                    found.append(os.path.dirname(res))
    return sorted(set(found))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='*', help='model folders relative to pca8/ (default: discover)')
    ap.add_argument('--include-ablation', action='store_true')
    ap.add_argument('--data-dir', default=DATA_DIR, help='data splits (for the PCA-only baseline only)')
    ap.add_argument('--prefix', default=DATA_PREFIX)
    a = ap.parse_args()
    globals()['DATA_DIR'], globals()['DATA_PREFIX'] = os.path.expanduser(a.data_dir), a.prefix
    models = [os.path.join(HERE, m) for m in a.models] if a.models else \
        find_models(DEFAULT_GROUPS + (['ablation_pca8'] if a.include_ablation else []))
    print(f'{len(models)} model folders:'); [print('  ' + os.path.relpath(m, HERE)) for m in models]
    results = [r for r in (run_model(m) for m in models) if r]
    if results:
        write_comparison(results)


if __name__ == '__main__':
    main()
