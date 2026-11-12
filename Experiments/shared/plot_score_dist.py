#!/usr/bin/env python3
"""plot_score_dist.py -- drift-score distributions per unseen family, known vs. drifting test samples.
Run from Experiments2 (via run_plot_score_dist.slurm):   python3 plot_score_dist.py
Rows: best CADE-mini, quantum (full objective), complex twin (third row only in the 3-row version).
Columns: EMBER, Drebin, LAMDA.  Per panel: 8 unseen families x (known, drifting) boxplots, three seeds pooled.
y = raw drift score as the percentile of the model's own training scores of the same task (0-100).
Red triangle: drifting median below known median.
Outputs in figures/:
  fig_score_dist.pdf (2 rows), fig_score_dist_3rows.pdf (3 rows)
  score_dist_stats.csv   box statistics per model/dataset/family/group (for a pgfplots version) + per-family ROC-AUC"""
import os, glob, json, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
DS = ['EMBER', 'Drebin', 'lamda']
DS_NAME = {'EMBER': 'EMBER', 'Drebin': 'Drebin', 'lamda': 'LAMDA'}
# best CADE-mini by mean ROC-AUC per dataset (main table): encoder on EMBER and Drebin, full on LAMDA
CADE = {'EMBER': ('CADE-mini, encoder', 'cade_pca8_p702/cade_enc_8_702'),
        'Drebin': ('CADE-mini, encoder', 'cade_pca8_p702/cade_enc_8_702'),
        'lamda': ('CADE-mini, full', 'cade_pca8_p702/cade_full_8_702')}
QF = ('Quantum (pair + prototype)', 'quantum_pca8/pennylane/quantum_hybrid_v7_p702')
TW = ('Twin, complex', 'twin_pca8_p702/classical_fidtwin_c_702')
ROWS = [('cade', '#1baf7a'), ('quantum', '#2a78d6'), ('twin', '#eb6834')]
TRAIN_NAMES = ['train_raw_scores.npy', 'train_scores.npy', 'train_raw_score.npy']
KNOWN_COL = '#b4b4b4'
warn = set()


def model_for(row, ds):
    return CADE[ds] if row == 'cade' else QF if row == 'quantum' else TW


def load_task(ds, folder, run, split):
    d = os.path.join(ROOT, ds, 'pca8', folder, 'results', f'run{run}', f'split{split}')
    f = os.path.join(d, 'ood_scores.csv')
    if not os.path.exists(f):
        return None
    with open(f) as fh:
        rows = [(float(r['raw_score']), int(float(r['is_ood']))) for r in csv.DictReader(fh)]
    s = np.array([r[0] for r in rows]); y = np.array([r[1] for r in rows])
    fam = '?'
    if os.path.exists(os.path.join(d, 'summary.json')):
        fam = str(json.load(open(os.path.join(d, 'summary.json'))).get('ood_family', '?'))
    ref = None
    for n in TRAIN_NAMES:
        if os.path.exists(os.path.join(d, n)):
            ref = np.asarray(np.load(os.path.join(d, n)), dtype=float).ravel(); break
    if ref is None and os.path.exists(os.path.join(d, 'states.npz')):
        ref = train_from_states(os.path.join(d, 'states.npz'), s, ds, folder)
    if ref is None:  # fallback: percentile of the known test scores of the same task
        warn.add(f'{ds}/{folder}'); ref = s[y == 0]
    return s, y, fam, np.sort(ref)


SCORE_FORMS = {'1-F': lambda F: 1 - F, 'sqrt(1-F)': lambda F: np.sqrt(np.clip(1 - F, 0, None)),
               '1-sqrt(F)': lambda F: 1 - np.sqrt(F)}
verified = {}


def train_from_states(path, s_csv, ds, folder):
    """Rebuild training drift scores from saved states: pure-state prototype = leading eigenvector of each known
    family's density matrix; score = fidelity distance to the nearest prototype. Used only if the same computation
    reproduces the test scores in ood_scores.csv."""
    z = np.load(path)
    ktr = [k for k in z.files if k.startswith('states_train')]; kte = [k for k in z.files if k.startswith('states_test')]
    if not ktr or 'y_train' not in z.files:
        return None
    Xtr, ytr = z[ktr[0]], z['y_train']
    Xtr = Xtr / np.linalg.norm(Xtr, axis=1, keepdims=True)
    P = []
    for c in np.unique(ytr):
        S = Xtr[ytr == c]; rho = S.T @ S.conj() / len(S)
        w, V = np.linalg.eigh(rho); P.append(V[:, -1])
    P = np.array(P)
    fid = lambda X: (np.abs((X / np.linalg.norm(X, axis=1, keepdims=True)) @ P.conj().T) ** 2).max(1)
    key = f'{ds}/{folder}'
    if key not in verified:  # check which score form reproduces ood_scores.csv, once per model folder
        verified[key] = None
        if kte and len(z[kte[0]]) == len(s_csv):
            Fte = fid(z[kte[0]])
            for name, f in SCORE_FORMS.items():
                err = np.max(np.abs(f(Fte) - s_csv))
                if err < 1e-4:
                    verified[key] = name; print(f'INFO: {key}: rebuilt test scores match ood_scores.csv ({name}, max err {err:.1e})'); break
            if verified[key] is None:
                from scipy.stats import spearmanr
                print(f'INFO: {key}: rebuilt test scores do NOT match ood_scores.csv '
                      f'(max err 1-F {np.max(np.abs(1 - Fte - s_csv)):.3g}, Spearman {spearmanr(1 - Fte, s_csv)[0]:.3f})')
    if verified[key] is None:
        return None
    return SCORE_FORMS[verified[key]](fid(Xtr))


def pct(s, ref):
    return 100.0 * np.searchsorted(ref, s, side='right') / len(ref)


def auc(s, y):
    from scipy.stats import rankdata
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1)


def collect(ds, folder):
    """per split: family name, pooled known percentiles, pooled drifting percentiles, AUC of pooled raw scores per seed"""
    out = []
    for sp in range(8):
        K, D, A, fam, top = [], [], [], None, []
        for run in range(3):
            t = load_task(ds, folder, run, sp)
            if t is None:
                continue
            s, y, f, ref = t; fam = f; p = pct(s, ref)
            K.append(p[y == 0]); D.append(p[y == 1]); A.append(auc(s, y)); top.append(np.mean(s[y == 1] > ref[-1]))
        if K:
            out.append(dict(split=sp, family=fam, known=np.concatenate(K), drift=np.concatenate(D),
                            auc=float(np.mean(A)), above_max=float(np.mean(top))))
    return out


def box_stats(v):
    q = np.percentile(v, [5, 25, 50, 75, 95])
    return dict(n=len(v), p5=q[0], q1=q[1], median=q[2], q3=q[3], p95=q[4], mean=float(np.mean(v)))


data, stats = {}, []
for row, _ in ROWS:
    for ds in DS:
        lab, folder = model_for(row, ds)
        data[(row, ds)] = collect(ds, folder)
        for c in data[(row, ds)]:
            for g in ('known', 'drift'):
                st = box_stats(c[g])
                stats.append(dict(model=lab, dataset=DS_NAME[ds], split=c['split'], family=c['family'], group=g,
                                  **{k: round(float(v), 2) for k, v in st.items()},
                                  family_roc_auc=round(c['auc'], 3), drift_above_train_max=round(c['above_max'], 3),
                                  flag_median_below=int(np.median(c['drift']) < np.median(c['known']))))

plt.rcParams.update({'font.size': 7, 'axes.titlesize': 8, 'pdf.fonttype': 42, 'axes.edgecolor': '#888888',
                     'xtick.color': '#444444', 'ytick.color': '#444444'})


def draw(nrows, fname):
    fig, axes = plt.subplots(nrows, 3, figsize=(7.2, 1.85 * nrows + 0.5), sharey=True, constrained_layout=True,
                             squeeze=False)
    for i, (row, col) in enumerate(ROWS[:nrows]):
        for j, ds in enumerate(DS):
            ax = axes[i, j]; cells = data[(row, ds)]; lab = model_for(row, ds)[0]
            if not cells:
                ax.set_title(f'{DS_NAME[ds]}: no data'); continue
            x = np.arange(len(cells))
            bk = ax.boxplot([c['known'] for c in cells], positions=x - 0.19, widths=0.32, whis=(5, 95),
                            showfliers=False, patch_artist=True)
            bd = ax.boxplot([c['drift'] for c in cells], positions=x + 0.19, widths=0.32, whis=(5, 95),
                            showfliers=False, patch_artist=True)
            for b, fc in ((bk, KNOWN_COL), (bd, col)):
                for p in b['boxes']:
                    p.set(facecolor=fc, edgecolor='#333333', linewidth=0.5)
                for k in ('whiskers', 'caps'):
                    for l in b[k]:
                        l.set(color='#555555', linewidth=0.5)
                for l in b['medians']:
                    l.set(color='#111111', linewidth=0.9)
            for xi, c in zip(x, cells):
                if np.median(c['drift']) < np.median(c['known']):
                    ax.plot(xi, 108, marker='v', color='#d62728', ms=4, clip_on=False)
                ax.text(xi, -14, f"{c['auc']:.2f}", ha='center', va='top', fontsize=5.5, color='#555555')
            ax.set_xticks(x); ax.set_xticklabels([c['family'] for c in cells], rotation=40, ha='right', fontsize=6)
            ax.tick_params(axis='x', pad=9)
            ax.set_ylim(-2, 112); ax.set_yticks([0, 25, 50, 75, 100]); ax.set_xlim(-0.6, len(cells) - 0.4)
            ax.grid(axis='y', color='#e6e6e6', lw=0.5); ax.set_axisbelow(True)
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)
            ax.set_title(f'{DS_NAME[ds]} -- {lab}', fontsize=7.5)
            if j == 0:
                ax.set_ylabel('Score percentile\n(own training scores)')
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    h = [Patch(facecolor=KNOWN_COL, edgecolor='#333333', lw=0.5, label='Known test samples')]
    h += [Patch(facecolor=c, edgecolor='#333333', lw=0.5, label=f'Drifting ({n})')
          for (n, c) in [('best CADE-mini', ROWS[0][1]), ('quantum, full obj.', ROWS[1][1]), ('complex twin', ROWS[2][1])][:nrows]]
    h += [Line2D([], [], marker='v', color='#d62728', ls='none', ms=4, label='Drifting median below known median')]
    fig.legend(handles=h, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=len(h), frameon=False, fontsize=6.5)
    fig.savefig(os.path.join(out, fname), bbox_inches='tight'); plt.close(fig)


out = os.path.join(ROOT, 'figures'); os.makedirs(out, exist_ok=True)
draw(2, 'fig_score_dist.pdf'); draw(3, 'fig_score_dist_3rows.pdf')
with open(os.path.join(out, 'score_dist_stats.csv'), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(stats[0])); w.writeheader(); w.writerows(stats)
for w_ in sorted(warn):
    print(f'WARNING: no training-score file in {w_}; used the known TEST scores as the percentile reference')
print(f"{'model':28s} {'dataset':7s} {'family':16s} {'AUC':>5s} {'med known':>9s} {'med drift':>9s} {'>train max':>10s} flag")
for r in stats:
    if r['group'] == 'drift':
        mk = [s for s in stats if s['model'] == r['model'] and s['dataset'] == r['dataset'] and s['split'] == r['split'] and s['group'] == 'known'][0]
        print(f"{r['model']:28s} {r['dataset']:7s} {r['family'][:16]:16s} {r['family_roc_auc']:5.2f} {mk['median']:9.1f} "
              f"{r['median']:9.1f} {r['drift_above_train_max']:10.2f} {'V' if r['flag_median_below'] else ''}")
print('Saved figures/fig_score_dist.pdf, figures/fig_score_dist_3rows.pdf, figures/score_dist_stats.csv')
