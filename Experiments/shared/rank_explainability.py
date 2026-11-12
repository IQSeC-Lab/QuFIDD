#!/usr/bin/env python3
"""rank_explainability.py -- rank the models by explanation fidelity under CADE's protocol (drift_analysis_cade results).
Run from Experiments2/:   python3 rank_explainability.py [--datasets EMBER Drebin] [--fpr 0.10] [--drift-only]
Reads <dataset>/pca8/drift_analysis_cade/<model>/results/split<s>/explanations.csv (per explained sample: d_orig, d_cade,
d_boundary, d_coin, d_gradient, lowerbound, k, is_drift) and the model's own ood_scores.csv (known-family test scores ->
FPR threshold and percentiles), and recomputes per split, per model:
  suff      sufficiency  = share of explained samples whose score falls below the FPR-f threshold after substituting the
                           CADE-selected features (CADE's Table-6 crossing, at a boundary that means the same for every
                           geometry)
  spec      specificity  = suff(CADE features) - suff(random features with the same k, 20 repetitions)
  k         sparsity     = mean number of selected features
  rel_d     relative distance after substitution, d'/d (within-model ratio)
  cbp       known-family percentile of the perturbed score (DREAM's CBP; 0.5 = looks like a typical known sample)
  pct_drop  percentile before - after
The random baseline is recomputed here with the model reloaded (same code as drift_analysis_cade), so the script needs the
same environment as that analysis (PennyLane for the quantum models). Ranks are computed per split (1 = best), averaged over
splits and datasets; paired Wilcoxon tests (over the splits of both datasets) compare every model with the quantum hybrid.
Writes explainability_ranking.txt, explainability_ranking.csv and Tables/table_explainability.tex."""
import os, sys, csv, glob, json, argparse, importlib.util
import numpy as np

ORDER = ['quantum_hybrid_v7_p702', 'quantum_hybrid_v7_p702_paironly', 'classical_fidtwin_c_702', 'classical_fidtwin_p_702',
         'classical_fidtwin_r_702', 'cade_full_8_702', 'cade_enc_8_702', 'cade_tuned_8_702', 'cade_deep_8_702']
NAMES = dict(zip(ORDER, ['Quantum hybrid', 'Quantum (pair loss only)', 'Twin, complex', 'Twin, plain', 'Twin, real',
                         'CADE-mini, full', 'CADE-mini, encoder', 'CADE-mini, tuned', 'CADE-mini, deep']))
TEX = dict(zip(ORDER, ['Quantum detector', 'Quantum detector (CADE objective)', 'Twin, complex', 'Twin, plain', 'Twin, real',
                       'CADE-mini, full', 'CADE-mini, encoder', 'CADE-mini, tuned', 'CADE-mini, deep']))
METRICS = [('suff', 'Sufficiency', 'max'), ('spec', 'Specificity', 'max'), ('k', 'Features $k$', 'min'),
           ('rel_d', "$d'/d$", 'min'), ('cbp', 'CBP', 'min')]
# CBP is reported as the raw percentile; the rank uses |CBP - 0.5| (a perturbed sample should look like a TYPICAL known sample)
RANDOM_TRY = 20


def load_module(path):
    spec = importlib.util.spec_from_file_location('m_' + os.path.basename(os.path.dirname(path)), path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def per_split(ds, model, split, fpr, drift_only, rng):
    import torch
    base = os.path.join(ds, 'pca8', 'drift_analysis_cade', model)
    f = os.path.join(base, 'results', f'split{split}', 'explanations.csv')
    if not os.path.exists(f):
        return None
    rows = list(csv.DictReader(open(f)))
    if drift_only:
        rows = [r for r in rows if r['is_drift'] == '1']
    rows = [r for r in rows if r['d_cade']]
    if len(rows) < 3:
        return None
    M = load_module(os.path.join(base, f'run_drift_{model}.py'))
    data_dir = M.DEFAULT_DATA; mres = os.path.normpath(os.path.join(base, M.DEFAULT_MODEL_RESULTS))
    data = M.load_split(data_dir, 'drebin_new', split)
    (inp, Xtr, Xte, embed, geom, anchors, Dtr, Dte, Etr, Ete), rel, _ = M.reload_model(data, os.path.join(mres, 'run0', f'split{split}'))
    ytr, is_ood = data['ytr'], data['is_ood']; raw = Dte.min(1)
    known = np.sort(raw[is_ood == 0]); thr = float(np.quantile(known, 1 - fpr))
    pct = lambda s: np.searchsorted(known, s) / len(known)
    idx = np.array([int(r['test_index']) for r in rows]); k = np.array([int(r['k']) for r in rows])
    C = anchors.shape[0]
    ref_idx = np.array([np.flatnonzero(ytr == c)[Dtr[ytr == c, c].argmin()] for c in range(C)])
    cstar = Dte[idx].argmin(1); x = Xte[idx]; xr = Xtr[torch.as_tensor(ref_idx[cstar])]
    xn, xrn = x.numpy(), xr.numpy()
    sf = np.load(os.path.join(base, 'results', f'split{split}', 'selected_features.npz'), allow_pickle=True)
    selmap = {int(t): np.asarray(v, int) for t, v in zip(sf['test_index'], sf['cade'])}      # column positions; [-1] = none
    feats_cade = [selmap[int(i)][selmap[int(i)] >= 0] for i in idx]

    def score_after(sel):
        Xp = np.stack([M.perturb(xn[i], sel[i], xrn[i]) for i in range(len(idx))])
        with torch.no_grad():
            return M.dist_all(embed(torch.tensor(Xp)), anchors, geom).numpy().min(1)
    s0 = raw[idx]; s_cade = score_after(feats_cade)
    suff = float((s_cade <= thr).mean()); cbp = float(pct(s_cade).mean())
    rel_d = float(np.mean([float(r['d_cade']) for r in rows]) / max(np.mean([float(r['d_orig']) for r in rows]), 1e-12))
    sr = []
    for _ in range(RANDOM_TRY):
        sr.append((score_after([rng.choice(xn.shape[1], k[i], replace=False) for i in range(len(idx))]) <= thr).mean())
    return {'dataset': ds, 'model': model, 'split': split, 'n': len(rows), 'suff': suff, 'suff_random': float(np.mean(sr)),
            'spec': suff - float(np.mean(sr)), 'k': float(k.mean()), 'rel_d': rel_d, 'cbp': cbp,
            'pct_drop': float((pct(s0) - pct(s_cade)).mean()), 'pct_before': float(pct(s0).mean()), 'reload_rel_diff': rel}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--datasets', nargs='+', default=['EMBER', 'Drebin', 'lamda'])
    ap.add_argument('--fpr', type=float, default=0.10); ap.add_argument('--drift-only', action='store_true')
    ap.add_argument('--out', default='.'); a = ap.parse_args()
    rng = np.random.default_rng(0); recs = []
    for ds in a.datasets:
        for m in ORDER:
            for s in range(8):
                r = per_split(ds, m, s, a.fpr, a.drift_only, rng)
                if r:
                    recs.append(r); print(f'{ds} {m} split {s}: suff {r["suff"]:.2f} (random {r["suff_random"]:.2f}) k {r["k"]:.0f} rel_d {r["rel_d"]:.3f} cbp {r["cbp"]:.3f}', flush=True)
    if not recs:
        print('no results found'); return
    models = [m for m in ORDER if any(r['model'] == m for r in recs)]
    # per-split ranks (only over the models present in that split)
    ranks = {m: {k: [] for k, _, _ in METRICS} for m in models}; paired = {m: {k: [] for k, _, _ in METRICS} for m in models}
    for ds in a.datasets:
        for s in range(8):
            R = {r['model']: r for r in recs if r['dataset'] == ds and r['split'] == s}
            if len(R) < 2:
                continue
            for key, _, best in METRICS:
                vals = {m: (abs(R[m][key] - 0.5) if key == 'cbp' else R[m][key]) for m in R}; order = sorted(vals, key=lambda m: -vals[m] if best == 'max' else vals[m])
                rk = {}
                for i, m in enumerate(order):                     # average ranks for ties
                    tied = [mm for mm in order if abs(vals[mm] - vals[m]) < 1e-12]; rk[m] = np.mean([order.index(t) + 1 for t in tied])
                for m in R:
                    ranks[m][key].append(rk[m])
                    if 'quantum_hybrid_v7_p702' in R:
                        paired[m][key].append((vals[m], vals['quantum_hybrid_v7_p702']))
    try:
        from scipy.stats import wilcoxon
    except Exception:
        wilcoxon = None
    L = [f'Explanation fidelity under CADE\'s protocol -- FPR budget {a.fpr:.0%}{" -- truly drifting samples only" if a.drift_only else " -- all explained (inspected) samples"}',
         f'datasets: {", ".join(a.datasets)}; {len(recs)} (dataset, model, split) records; ranks per split (1 = best), mean over splits', '',
         f"{'model':22s} {'n':>5} " + ' '.join(f'{lab:>12s}' for _, lab, _ in METRICS) + f" {'rand suff':>9} {'mean rank':>9}"]
    out_rows = []
    for m in models:
        R = [r for r in recs if r['model'] == m]; mv = {k: float(np.mean([r[k] for r in R])) for k, _, _ in METRICS}
        mr = {k: float(np.mean(ranks[m][k])) if ranks[m][k] else float('nan') for k, _, _ in METRICS}
        comp = float(np.nanmean(list(mr.values())))
        L.append(f'{NAMES[m]:22s} {sum(r["n"] for r in R):5d} ' + ' '.join(f'{mv[k]:7.3f}({mr[k]:4.1f})' for k, _, _ in METRICS) +
                 f' {np.mean([r["suff_random"] for r in R]):9.3f} {comp:9.2f}')
        out_rows.append({'model': m, **{k: mv[k] for k in mv}, **{f'rank_{k}': mr[k] for k in mr}, 'mean_rank': comp,
                         'suff_random': float(np.mean([r['suff_random'] for r in R])), 'n_records': len(R)})
    L += ['', 'value(rank): Sufficiency = share of explained samples no longer flagged after substituting the CADE-selected features;',
          'Specificity = sufficiency minus that of random features with the same k; k = selected features; d\'/d = distance after / before;',
          'CBP = known-family percentile of the perturbed score (0.5 = typical known sample; ranked by |CBP - 0.5|). Higher is better for the first two, lower for k and d\'/d.']
    if wilcoxon:
        L += ['', 'Paired Wilcoxon (over splits x datasets) of each model vs. the quantum hybrid; + = model better than quantum:']
        for m in models:
            if m == 'quantum_hybrid_v7_p702':
                continue
            parts = []
            for key, lab, best in METRICS:
                pr = paired[m][key]
                if len(pr) < 6:
                    parts.append(f'{lab}: n/a'); continue
                x_, y_ = np.array([p[0] for p in pr]), np.array([p[1] for p in pr]); d = (x_ - y_) if best == 'max' else (y_ - x_)
                try:
                    p = wilcoxon(x_, y_).pvalue if np.any(x_ != y_) else 1.0
                except Exception:
                    p = float('nan')
                parts.append(f'{lab}: {np.mean(d):+.3f} (p={p:.3f})')
            L.append(f'  {NAMES[m]:22s} ' + '; '.join(parts))
    txt = '\n'.join(L); print(txt)
    sfx = f'_fpr{int(100 * a.fpr)}{"_driftonly" if a.drift_only else ""}'
    open(os.path.join(a.out, f'explainability_ranking{sfx}.txt'), 'w').write(txt + '\n')
    with open(os.path.join(a.out, f'explainability_ranking{sfx}.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0])); w.writeheader(); w.writerows(recs)
    # LaTeX table
    os.makedirs(os.path.join(a.out, 'Tables'), exist_ok=True)
    best = {k: (max if b == 'max' else min)((abs(r[k] - 0.5) if k == 'cbp' else r[k]) for r in out_rows) for k, _, b in METRICS}
    T = ['% Requires \\usepackage{booktabs}', '\\begin{table}[t]', '  \\centering',
         f'  \\caption{{Explanation fidelity under CADE\'s protocol (mean over the splits of {" and ".join(a.datasets)}; per-split rank in parentheses, 1 = best). '
         f'Sufficiency is the share of explained samples that are no longer flagged at the {a.fpr:.0%}-FPR boundary after substituting the features selected by CADE\'s explainer; '
         'specificity subtracts the same quantity for random features with the same $k$; $d\'/d$ is the distance to the nearest family after substitution relative to before; '
         'CBP is the known-family percentile of the perturbed score (0.5 is a typical known sample; ranked by its distance to 0.5). Bold marks the best value per column.}',
         '  \\label{tab:explainability}', '  \\small', '  \\setlength{\\tabcolsep}{3pt}', '  \\begin{tabular}{@{}l' + 'c' * len(METRICS) + 'c@{}}', '    \\toprule',
         '    Model & ' + ' & '.join(lab for _, lab, _ in METRICS) + ' & Mean rank \\\\', '    \\midrule']
    grp_break = {'quantum_hybrid_v7_p702_paironly': False, 'classical_fidtwin_c_702': True, 'cade_full_8_702': True}
    for r in out_rows:
        if grp_break.get(r['model']):
            T.append('    \\midrule')
        cells = []
        for k, _, _ in METRICS:
            v = f'{r[k]:.1f}' if k == 'k' else f'{r[k]:.3f}'
            v = f'\\textbf{{{v}}}' if abs((abs(r[k] - 0.5) if k == 'cbp' else r[k]) - best[k]) < 1e-12 else v
            cells.append(f'{v} ({r[f"rank_{k}"]:.1f})')
        T.append(f'    {TEX[r["model"]]} & ' + ' & '.join(cells) + f' & {r["mean_rank"]:.2f} \\\\')
    T += ['    \\bottomrule', '  \\end{tabular}', '\\end{table}']
    open(os.path.join(a.out, 'Tables', f'table_explainability{sfx}.tex'), 'w').write('\n'.join(T) + '\n')
    print(f'\nwrote explainability_ranking{sfx}.txt/.csv and Tables/table_explainability{sfx}.tex')


if __name__ == '__main__':
    main()
