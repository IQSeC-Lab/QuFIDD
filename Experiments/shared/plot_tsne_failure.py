#!/usr/bin/env python3
"""plot_tsne_failure.py -- t-SNE of the quantum detector (full objective) in its weakest settings, with the fidelity
distance 1-F (precomputed). Run from Experiments2 via SLURM:  python3 plot_tsne_failure.py [--run 0]
Panels: EMBER/Xtrat, Drebin/FakeInstaller, LAMDA/Plankton (+ LAMDA/FakeInst on a second page).
Marks: known test samples coloured by family; drifting samples as x -- black if detected, red if MISSED (score below the
median known score); known samples above the ranked-F1 stopping point (false positives) ringed; prototypes as stars.
Writes figures/fig_tsne_failure.pdf and figures/tsne_failure_coords.csv."""
import os, csv, json, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

ap = argparse.ArgumentParser(); ap.add_argument('--run', type=int, default=0)
ap.add_argument('--model', default='quantum_pca8/pennylane/quantum_hybrid_v7_p702')
ap.add_argument('--n-known', type=int, default=1500); ap.add_argument('--n-drift', type=int, default=600)
ap.add_argument('--n-train', type=int, default=800)
a = ap.parse_args()
NAMES = {'EMBER': ['Xtrat', 'Zbot', 'Ramnit', 'Sality', 'Wapomi', 'Lethic', 'Ursnif', 'Kovter'],
         'Drebin': ['FakeInstaller', 'DroidKungFu', 'Plankton', 'GingerMaster', 'BaseBridge', 'Iconosys', 'Kmin', 'FakeDoc'],
         'lamda': ['DroidKungFu', 'Plankton', 'BaseBridge', 'SMSreg', 'Airpush', 'FakeInst', 'Shedun', 'SMSsend']}
SETTINGS = [('EMBER', 0), ('Drebin', 0), ('lamda', 1), ('lamda', 5)]   # family id = split index (verified Oct 29 2026)
PAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#8e5bd0', '#d6457a', '#5a8f29', '#8c6d31']
plt.rcParams.update({'font.size': 7, 'pdf.fonttype': 42, 'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix'})
rng = np.random.default_rng(0)
results, rows_out = [], []
for ds, sp in SETTINGS:
    d = f'{ds}/pca8/{a.model}/results/run{a.run}/split{sp}'
    z = np.load(f'{d}/states.npz')
    Str = z[[k for k in z.files if k.startswith('states_train')][0]]; Ste = z[[k for k in z.files if k.startswith('states_test')][0]]
    ytr, yte = z['y_train'], z['y_test']
    Str = Str / np.linalg.norm(Str, axis=1, keepdims=True); Ste = Ste / np.linalg.norm(Ste, axis=1, keepdims=True)
    labs = np.unique(ytr); P = []
    for c in labs:
        S = Str[ytr == c]; w, V = np.linalg.eigh(S.T @ S.conj() / len(S)); P.append(V[:, -1])
    P = np.array(P)
    rr = list(csv.DictReader(open(f'{d}/ood_scores.csv')))
    s = np.array([float(r['raw_score']) for r in rr]); ood = np.array([int(float(r['is_ood'])) for r in rr])
    err = np.max(np.abs(1 - (np.abs(Ste @ P.conj().T) ** 2).max(1) - s))
    known = ood == 0; held = set(int(v) for v in np.unique(yte[~known]))
    if set(int(v) for v in np.unique(yte[known])) <= set(int(c) for c in labs) and not (held & set(int(c) for c in labs)):
        proto_orig = [int(c) for c in labs]; tr_orig = ytr.astype(int)
    else:
        proto_orig = [c for c in range(len(labs) + len(held)) if c not in held]; tr_orig = np.array(proto_orig)[np.searchsorted(labs, ytr)]
    # missed drift and false positives (as in failure_analysis.py)
    missed = (~known) & (s < np.median(s[known]))
    o = np.argsort(-s, kind='stable'); tp = np.cumsum(ood[o]); k = np.arange(1, len(s) + 1)
    stop = int(np.argmax(2 * tp / (k + ood.sum()))) + 1; fp = np.zeros(len(s), bool); fp[o[:stop]] = True; fp &= known
    # subsample (same rule for every panel)
    ik = rng.choice(np.where(known)[0], min(a.n_known, known.sum()), replace=False)
    idr = rng.choice(np.where(~known)[0], min(a.n_drift, (~known).sum()), replace=False)
    itr = rng.choice(len(ytr), min(a.n_train, len(ytr)), replace=False)
    X = np.r_[Str[itr], Ste[ik], Ste[idr], P]
    kind = np.r_[np.zeros(len(itr), int), np.ones(len(ik), int), 2 * np.ones(len(idr), int), 3 * np.ones(len(P), int)]
    fam = np.r_[tr_orig[itr], yte[ik].astype(int), yte[idr].astype(int), proto_orig]
    flag = np.r_[np.zeros(len(itr), bool), fp[ik], missed[idr], np.zeros(len(P), bool)]
    D = np.clip(1 - np.abs(X @ X.conj().T) ** 2, 0, None); D = (D + D.T) / 2; np.fill_diagonal(D, 0)
    Y = TSNE(n_components=2, metric='precomputed', init='random', perplexity=30, random_state=0).fit_transform(D)
    auc_ = None
    from scipy.stats import rankdata
    r_ = rankdata(s); n1 = ood.sum(); auc_ = (r_[ood == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(s) - n1))
    print(f'{ds:6s} unseen {NAMES[ds][sp]:13s} run {a.run}: score check {err:.1e}, ROC-AUC {auc_:.3f}, missed {100 * missed[~known].mean():.0f}% of drift, '
          f'FPs {fp.sum()} ({100 * fp.sum() / known.sum():.0f}% of known): ' +
          ', '.join(f'{NAMES[ds][c]} {n}' for c, n in sorted(((int(c), int((yte[fp] == c).sum())) for c in np.unique(yte[fp])), key=lambda t: -t[1])))
    results.append((ds, sp, Y, kind, fam, flag, auc_, sorted(set(proto_orig))))
    for i in range(len(Y)):
        rows_out.append([ds, NAMES[ds][sp], round(float(Y[i, 0]), 3), round(float(Y[i, 1]), 3), ['train', 'known', 'drift', 'proto'][kind[i]],
                         NAMES[ds][fam[i]], int(flag[i])])


def panel(ax, res):
    ds, sp, Y, kind, fam, flag, auc_, fams = res
    col = {c: PAL[i % len(PAL)] for i, c in enumerate(fams)}
    m = kind == 0; ax.scatter(Y[m, 0], Y[m, 1], s=1.5, c='#d8d8d8', lw=0, rasterized=True)
    for c in fams:
        m = (kind == 1) & (fam == c); ax.scatter(Y[m, 0], Y[m, 1], s=3, c=col[c], alpha=0.55, lw=0, rasterized=True)
    m = (kind == 1) & flag; ax.scatter(Y[m, 0], Y[m, 1], s=11, facecolors='none', edgecolors='black', linewidths=0.35, rasterized=True)
    m = (kind == 2) & ~flag; ax.scatter(Y[m, 0], Y[m, 1], s=6, c='black', marker='x', lw=0.4, rasterized=True)
    m = (kind == 2) & flag; ax.scatter(Y[m, 0], Y[m, 1], s=7, c='#d62728', marker='x', lw=0.6, rasterized=True)
    for c in fams:
        m = (kind == 3) & (fam == c); ax.scatter(Y[m, 0], Y[m, 1], s=70, c=col[c], marker='*', edgecolors='black', linewidths=0.6, zorder=6)
    ax.set_xticks([]); ax.set_yticks([])
    for sp_ in ax.spines.values():
        sp_.set_color('#bbbbbb')
    dn = {'EMBER': 'EMBER', 'Drebin': 'Drebin', 'lamda': 'LAMDA'}[ds]
    ax.set_title(f'{dn}, unseen {NAMES[ds][sp]} (ROC-AUC {auc_:.2f})', fontsize=7.5)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker='o', ls='none', ms=3.5, color=col[c], label=NAMES[ds][c]) for c in fams]
    ax.legend(handles=h, loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=4, frameon=False, fontsize=5.8,
              handletextpad=0.1, columnspacing=0.6)


from matplotlib.lines import Line2D
common = [Line2D([], [], marker='x', ls='none', ms=4, color='black', label='Drifting, detected'),
          Line2D([], [], marker='x', ls='none', ms=4, color='#d62728', label='Drifting, missed (below median known score)'),
          Line2D([], [], marker='o', ls='none', ms=4, markerfacecolor='none', markeredgecolor='black', label='Known, false positive at ranked-F1 stop'),
          Line2D([], [], marker='*', ls='none', ms=7, color='#888888', markeredgecolor='black', label='Prototype'),
          Line2D([], [], marker='o', ls='none', ms=3, color='#d8d8d8', label='Training')]
out = 'figures'; os.makedirs(out, exist_ok=True)
for fname, sel in (('fig_tsne_failure.pdf', results[:3]), ('fig_tsne_failure_lamda_fakeinst.pdf', results[3:])):
    if not sel:
        continue
    fig, axes = plt.subplots(1, len(sel), figsize=(2.45 * len(sel), 2.75), squeeze=False)
    for ax, res in zip(axes[0], sel):
        panel(ax, res)
    fig.legend(handles=common, loc='lower center', bbox_to_anchor=(0.5, 0.98), ncol=3, frameon=False, fontsize=6.5)
    fig.tight_layout(); fig.savefig(os.path.join(out, fname), bbox_inches='tight', dpi=300); plt.close(fig)
with open(os.path.join(out, 'tsne_failure_coords.csv'), 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['dataset', 'unseen', 'x', 'y', 'kind', 'family', 'flag_missed_or_fp']); w.writerows(rows_out)
print('Saved figures/fig_tsne_failure.pdf, figures/fig_tsne_failure_lamda_fakeinst.pdf, figures/tsne_failure_coords.csv')
