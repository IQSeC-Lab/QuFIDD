#!/usr/bin/env python3
"""
analyze_drift_type.py -- capacity_matched / capacity_matched_8fam (top level, next to evaluate.py)
======================================================================================================================
EXPLANATORY analysis (uses the test labels to EXPLAIN results, never to select anything): does the TYPE of drift of
each held-out family explain which quantum encoding detects it, and which ensemble weight it prefers?

Hypothesis. Both encodings start from the same PCA-8 vector z (centred by the training mean):
  amplitude encoding keeps only the DIRECTION z / ||z||  -> blind to magnitude drift
  angle encoding keeps each PC's VALUE (arctan of the centred, scaled PC) -> sees magnitude, warps direction
so families that differ from the known ones mainly in DIRECTION should favour amplitude, families that differ mainly
in MAGNITUDE should favour angle, and families that differ in both should favour the mix.

Per drift family (one split = one held-out family; PCA-8 fitted on that split's training data, no model involved):
  direction novelty   ROC-AUC (known test vs drift) of the cosine distance to the nearest known-family mean direction
  magnitude novelty   ROC-AUC of | log||z|| - median_c | / MAD_c, with c the nearest known family (by direction) and
                      median_c / MAD_c of log||z|| over that family's training samples (two-sided: longer OR shorter)
  norm shift          median log||z|| of the drift samples minus that of the known test samples (> 0: drift is longer)
compared with, from the saved results:
  amp-8 / angle-8 ROC-AUC (mean over seeds) and their difference, and the best ensemble weight w for that family from
  pca8/comparison/ensemble_weight_curve.csv (w = 0: amplitude only, w = 1: angle only; run ensemble_angle_amp.py first)
Prediction: (direction - magnitude novelty) correlates POSITIVELY with (amp - angle ROC) and NEGATIVELY with best w.

Usage (from capacity_matched/ or capacity_matched_8fam/):   python3 analyze_drift_type.py pca8
Output: pca8/comparison/drift_type.txt, drift_type.csv, plots/drift_type.png
"""
import os, sys, glob, json, csv, argparse, importlib.util
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr


def load_module(root):
    path = sorted(glob.glob(os.path.join(root, 'quantum_amp_*', 'run_quantum_amp_*.py')))[0]
    spec = importlib.util.spec_from_file_location('base', path); m = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, [path]; spec.loader.exec_module(m); sys.argv = argv
    return m


def mad_stats(x):
    m = float(np.median(x)); s = 1.4826 * float(np.median(np.abs(x - m)))
    return m, (s if s > 0 else 1e-12)


def model_roc(root, prefix):
    out = {}
    for p in glob.glob(os.path.join(root, f'{prefix}*', 'results', 'run*', 'split*', 'summary.json')):
        d = json.load(open(p)); out.setdefault(d['ood_family'], []).append(d['evaluation']['threshold_free']['ROC_AUC'])
    return {f: float(np.mean(v)) for f, v in out.items()}


def best_w(root):
    p = os.path.join(root, 'comparison', 'ensemble_weight_curve.csv')
    if not os.path.exists(p):
        return {}, {}
    agg = {}
    for r in csv.DictReader(open(p)):
        if r['pair'].startswith('Q:'):
            agg.setdefault((r['family'], float(r['w'])), []).append(float(r['ROC']))
    fams = sorted({f for f, _ in agg}); bw, peak = {}, {}
    for f in fams:
        ws = sorted(w for ff, w in agg if ff == f); v = [np.mean(agg[(f, w)]) for w in ws]
        bw[f] = ws[int(np.argmax(v))]; peak[f] = float(max(v))
    return bw, peak


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('pca_dir'); ap.add_argument('--data-dir', default=None)
    a = ap.parse_args(); root = os.path.abspath(a.pca_dir); mod = load_module(root)
    data_dir = os.path.expanduser(a.data_dir or mod.DEFAULT_DATA)
    amp, ang = model_roc(root, 'quantum_amp_'), model_roc(root, 'quantum_angle_'); bw, peak = best_w(root)
    rows, lines = [], []
    def log(s=''):
        print(s, flush=True); lines.append(s)
    log('=' * 116); log(f'DRIFT TYPE vs ENCODING -- {root}   (explanatory; test labels used to explain, not to select)'); log('=' * 116)
    for s in range(mod.N_SPLITS):
        data = mod.load_split(data_dir, mod.DEFAULT_PREFIX, s); pca = mod.fit_pca(data['Xtr'])
        Ztr, Zte = pca.transform(data['Xtr']), pca.transform(data['Xte']); ytr, ood = data['ytr'], data['is_ood']
        C = len(np.unique(ytr)); unit = lambda Z: Z / np.linalg.norm(Z, axis=1, keepdims=True).clip(min=1e-12)
        Utr, Ute = unit(Ztr), unit(Zte)
        mu = np.stack([unit(Utr[ytr == c].mean(0, keepdims=True))[0] for c in range(C)])
        cos = Ute @ mu.T; near = cos.argmax(1); dir_score = 1 - cos.max(1)
        ln_tr, ln_te = np.log(np.linalg.norm(Ztr, axis=1) + 1e-12), np.log(np.linalg.norm(Zte, axis=1) + 1e-12)
        stats = [mad_stats(ln_tr[ytr == c]) for c in range(C)]
        mag_score = np.array([abs(ln_te[i] - stats[near[i]][0]) / stats[near[i]][1] for i in range(len(ln_te))])
        fam = data['family']
        row = {'family': fam, 'dir_ROC': roc_auc_score(ood, dir_score), 'mag_ROC': roc_auc_score(ood, mag_score),
               'norm_shift': float(np.median(ln_te[ood == 1]) - np.median(ln_te[ood == 0])),
               'amp_ROC': amp.get(fam, np.nan), 'angle_ROC': ang.get(fam, np.nan),
               'best_w': bw.get(fam, np.nan), 'peak_ROC': peak.get(fam, np.nan)}
        row['dir_minus_mag'] = row['dir_ROC'] - row['mag_ROC']; row['amp_minus_angle'] = row['amp_ROC'] - row['angle_ROC']
        rows.append(row)
    cols = [('family', '<14', 's'), ('dir_ROC', '>9', '.3f'), ('mag_ROC', '>9', '.3f'), ('dir_minus_mag', '>10', '+.3f'),
            ('norm_shift', '>11', '+.3f'), ('amp_ROC', '>9', '.3f'), ('angle_ROC', '>10', '.3f'),
            ('amp_minus_angle', '>12', '+.3f'), ('best_w', '>8', '.1f')]
    log('\n  direction / magnitude novelty: ROC-AUC of each input-space score alone (PCA-8, no model); norm shift: drift')
    log('  minus known median log-length (> 0 = drift vectors are longer); best w: 0 = amplitude only, 1 = angle only\n')
    log('  ' + ''.join(f'{"family" if c == "family" else c.replace("_minus_", "-").replace("_ROC", ""):{al}}' for c, al, _ in cols))
    def cell(v, al, fm):
        txt = 'n/a' if (isinstance(v, float) and np.isnan(v)) else (v if fm == 's' else format(v, fm))
        return f'{txt:{al}}'
    for r in sorted(rows, key=lambda r: -r['dir_minus_mag']):
        log('  ' + ''.join(cell(r[c], al, fm) for c, al, fm in cols))
    log('\n  Spearman correlations across families (n = %d; small n -> read the sign and size, not just p):' % len(rows))
    for x, y, exp in (('dir_minus_mag', 'amp_minus_angle', '+'), ('dir_minus_mag', 'best_w', '-'),
                      ('norm_shift', 'amp_minus_angle', '?'), ('mag_ROC', 'angle_ROC', '+'), ('dir_ROC', 'amp_ROC', '+')):
        v = [(r[x], r[y]) for r in rows if not np.isnan(r[x]) and not np.isnan(r[y])]
        if len(v) >= 4 and (len({a for a, _ in v}) == 1 or len({b for _, b in v}) == 1):
            log(f'    {x:<15} vs {y:<16} undefined (one of the two is the same for every family)')
        elif len(v) >= 4:
            rho, p = spearmanr([a for a, _ in v], [b for _, b in v])
            log(f'    {x:<15} vs {y:<16} rho = {rho:+.2f}  (p = {p:.3f})   predicted sign: {exp}')
        else:
            log(f'    {x:<15} vs {y:<16} not enough families with both values'
                + (' (best_w needs pca8/comparison/ensemble_weight_curve.csv from ensemble_angle_amp.py)' if y == 'best_w' else ''))
    out = os.path.join(root, 'comparison'); os.makedirs(os.path.join(out, 'plots'), exist_ok=True)
    open(os.path.join(out, 'drift_type.txt'), 'w').write('\n'.join(lines) + '\n')
    with open(os.path.join(out, 'drift_type.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
        for a_, y, lab in ((ax[0], 'amp_minus_angle', 'amp-8 ROC - angle-8 ROC'), (ax[1], 'best_w', 'best ensemble weight w (1 = angle)')):
            pts = [r for r in rows if not np.isnan(r[y])]
            a_.scatter([r['dir_minus_mag'] for r in pts], [r[y] for r in pts])
            for r in pts:
                a_.annotate(r['family'], (r['dir_minus_mag'], r[y]), fontsize=8, xytext=(3, 3), textcoords='offset points')
            a_.axhline(0 if y == 'amp_minus_angle' else 0.5, color='grey', lw=0.8, ls='--'); a_.axvline(0, color='grey', lw=0.8, ls='--')
            a_.set_xlabel('direction novelty - magnitude novelty (input ROC-AUC)'); a_.set_ylabel(lab); a_.grid(alpha=0.3)
        fig.suptitle('Drift type vs encoding preference (explanatory)', fontsize=10); fig.tight_layout()
        fig.savefig(os.path.join(out, 'plots', 'drift_type.png'), dpi=150); plt.close(fig)
    except Exception as ex:
        log(f'  (plot skipped: {ex})')
    print(f'\nSaved: {out}/drift_type.txt, drift_type.csv, plots/drift_type.png')


if __name__ == '__main__':
    main()
