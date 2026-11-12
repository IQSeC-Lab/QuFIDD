#!/usr/bin/env python3
"""plot_tsne.py -- t-SNE of the PCA-8 input and the learned embeddings for ONE setting (default: EMBER, split 1 = Zbot,
run 2 = seed 456). Run from Experiments2 (via run_plot_tsne.slurm):
    python3 plot_tsne.py [--ds EMBER] [--split 1] [--run 2]
Panels: (1) PCA-8 input, Euclidean; (2) CADE-mini encoder latent, Euclidean (recomputed from model_weights.pt and checked
against ood_scores.csv); (3) complex twin states, fidelity distance 1-F; (4) quantum states (pair + prototype), 1-F.
Same samples in every panel: a subsample of known test samples, drifting samples, and training samples, plus centroids
(panels 1-2) or pure-state prototypes (panels 3-4) included in the distance matrix before t-SNE.
Outputs in figures/: fig_tsne_<ds>_<family>.pdf, tsne_<ds>_<family>_coords.csv (for a pgfplots version),
and a printed table with the per-panel ROC-AUC and the kNN absorption rate in the ORIGINAL distance of each panel."""
import os, csv, json, argparse, importlib.util
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from sklearn.manifold import TSNE

ROOT = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--ds', default='EMBER'); ap.add_argument('--split', type=int, default=1); ap.add_argument('--run', type=int, default=2)
ap.add_argument('--n-known', type=int, default=1500); ap.add_argument('--n-drift', type=int, default=600)
ap.add_argument('--n-train', type=int, default=1000); ap.add_argument('--perplexity', type=float, default=30)
ap.add_argument('--cade', default='cade_pca8_p702/cade_enc_8_702'); ap.add_argument('--cade-hidden', default='44')
a = ap.parse_args()
base = os.path.join(ROOT, a.ds, 'pca8')
res = lambda m: os.path.join(base, m, 'results', f'run{a.run}', f'split{a.split}')
TWIN, QUANT = 'twin_pca8_p702/classical_fidtwin_c_702', 'quantum_pca8/pennylane/quantum_hybrid_v7_p702'


def auc(s, y):
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def csv_scores(d):
    rows = list(csv.DictReader(open(os.path.join(d, 'ood_scores.csv'))))
    return np.array([float(r['raw_score']) for r in rows]), np.array([int(float(r['is_ood'])) for r in rows])


# ---------------------------------------------------------------- data, PCA and CADE latents via the run script itself
cade_dir = os.path.join(base, a.cade)
script = [f for f in os.listdir(cade_dir) if f.startswith('run_') and f.endswith('.py')][0]
spec = importlib.util.spec_from_file_location('cade_run', os.path.join(cade_dir, script)); C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)
data = C.load_split(C.DEFAULT_DATA, C.DEFAULT_PREFIX, a.split)
pca = C.fit_pca(data['Xtr'])
Ztr = pca.transform(data['Xtr']).astype(np.float32); Zte = pca.transform(data['Xte']).astype(np.float32)
ytr, yte, is_ood = data['ytr'], data['yte'], data['is_ood']
known_labels = data['known_labels']
print(f'{a.ds} split {a.split} run {a.run}: unseen family {data["family"]}, train {len(ytr)}, test {len(yte)} ({is_ood.sum()} drifting)')

import torch, torch.nn as nn
dims = [C.PCA_DIM] + [int(h) for h in a.cade_hidden.split(',') if h] + [len(np.unique(ytr))]
enc = []
for i in range(len(dims) - 2):
    enc += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
enc += [nn.Linear(dims[-2], dims[-1])]
encoder = nn.Sequential(*enc)
sd = torch.load(os.path.join(res(a.cade), 'model_weights.pt'), map_location='cpu')
mods = nn.ModuleList([encoder]); mods.load_state_dict({k: v for k, v in sd.items() if k.startswith('0.')})
with torch.no_grad():
    Ltr = encoder(torch.from_numpy(Ztr)).numpy(); Lte = encoder(torch.from_numpy(Zte)).numpy()
s_c, y_c = csv_scores(res(a.cade))
cent = np.array([Ltr[ytr == c].mean(0) for c in range(len(known_labels))])
s_re = np.linalg.norm(Lte[:, None] - cent[None], axis=2).min(1)
err = np.max(np.abs(s_re - s_c)) if len(s_re) == len(s_c) else np.inf
print(f'CHECK CADE latents: max |recomputed - ood_scores.csv| = {err:.2e}  ' + ('OK' if err < 1e-3 else '!! MISMATCH -- panel 2 not trustworthy'))


# ---------------------------------------------------------------- fidelity states
def states(m):
    z = np.load(os.path.join(res(m), 'states.npz'))
    ktr = [k for k in z.files if k.startswith('states_train')][0]; kte = [k for k in z.files if k.startswith('states_test')][0]
    Str, Ste = z[ktr], z[kte]
    Str = Str / np.linalg.norm(Str, axis=1, keepdims=True); Ste = Ste / np.linalg.norm(Ste, axis=1, keepdims=True)
    yy = z['y_train'] if 'y_train' in z.files else ytr
    if len(Str) != len(ytr) or len(Ste) != len(yte):
        print(f'!! {m}: state counts {len(Str)}/{len(Ste)} differ from data {len(ytr)}/{len(yte)}')
    labs = np.unique(yy); P = []
    for c in labs:
        S = Str[yy == c]; w, V = np.linalg.eigh(S.T @ S.conj() / len(S)); P.append(V[:, -1])
    P = np.array(P)
    s, _ = csv_scores(res(m)); F = (np.abs(Ste @ P.conj().T) ** 2).max(1)
    print(f'CHECK {m.split("/")[-1]} states: max |(1-F) - ood_scores.csv| = {np.max(np.abs(1 - F - s)):.2e}')
    return Str, Ste, P


Ttr, Tte, TP = states(TWIN)
Qtr, Qte, QP = states(QUANT)

# ---------------------------------------------------------------- common subsample (same points in every panel)
rng = np.random.default_rng(0)
ik = rng.choice(np.where(is_ood == 0)[0], min(a.n_known, (is_ood == 0).sum()), replace=False)
idr = rng.choice(np.where(is_ood == 1)[0], min(a.n_drift, is_ood.sum()), replace=False)
itr = rng.choice(len(ytr), min(a.n_train, len(ytr)), replace=False)
te_idx = np.r_[ik, idr]; K = len(known_labels)
kind = np.r_[np.zeros(len(itr), int), np.ones(len(ik), int), 2 * np.ones(len(idr), int), 3 * np.ones(K, int)]  # 0 train 1 known 2 drift 3 star
fam = np.r_[ytr[itr], [known_labels.index(int(v)) if int(v) in known_labels else -1 for v in yte[te_idx]], np.arange(K)]


def euclid(Xtr, Xte, cen):
    X = np.r_[Xtr[itr], Xte[te_idx], cen]
    sq = (X ** 2).sum(1); D = np.sqrt(np.clip(sq[:, None] + sq[None] - 2 * X @ X.T, 0, None)); return D


def fiddist(Str, Ste, P):
    X = np.r_[Str[itr], Ste[te_idx], P]
    return np.clip(1 - np.abs(X @ X.conj().T) ** 2, 0, None)


pca_cent = np.array([Ztr[ytr == c].mean(0) for c in range(K)])
s_pca = np.linalg.norm(Zte[:, None] - pca_cent[None], axis=2).min(1)
PANELS = [('PCA-8 input', euclid(Ztr, Zte, pca_cent), auc(s_pca, is_ood)),
          ('CADE-mini (encoder)', euclid(Ltr, Lte, cent), auc(s_c, y_c)),
          ('Twin, complex', fiddist(Ttr, Tte, TP), auc(*csv_scores(res(TWIN)))),
          ('Quantum (pair + prototype)', fiddist(Qtr, Qte, QP), auc(*csv_scores(res(QUANT))))]


def absorption(D, k=10):
    """share of drifting samples whose k nearest TEST neighbours (original distance) are mostly known samples"""
    t = np.where((kind == 1) | (kind == 2))[0]; Dt = D[np.ix_(t, t)].copy(); np.fill_diagonal(Dt, np.inf)
    nn_ = np.argsort(Dt, 1)[:, :k]; isd = kind[t] == 2
    return float(np.mean((~isd[nn_]).mean(1)[isd] > 0.5))


def nearest_euclid(X, cen):
    return np.linalg.norm(X[:, None] - cen[None], axis=2).min(1)


def nearest_fid(S, P):
    return 1 - (np.abs(S @ P.conj().T) ** 2).max(1)


SC = [(nearest_euclid(Ztr, pca_cent), nearest_euclid(Zte, pca_cent)), (nearest_euclid(Ltr, cent), nearest_euclid(Lte, cent)),
      (nearest_fid(Ttr, TP), nearest_fid(Tte, TP)), (nearest_fid(Qtr, QP), nearest_fid(Qte, QP))]
PCT = []
for str_, ste in SC:
    ref = np.sort(str_); pc = lambda v: 100.0 * np.searchsorted(ref, v, side='right') / len(ref)
    PCT.append(np.r_[pc(str_[itr]), pc(ste[te_idx]), np.full(K, np.nan)])

coords, out_rows = [], []
for name, D, A in PANELS:
    D = (D + D.T) / 2; np.fill_diagonal(D, 0)
    Y = TSNE(n_components=2, metric='precomputed', init='random', perplexity=a.perplexity, random_state=0).fit_transform(D)
    coords.append(Y); ab = absorption(D)
    pc = PCT[len(coords) - 1]
    out_rows.append((name, A, ab)); print(f'{name:28s} ROC-AUC {A:.3f}   drifting samples with mostly-known 10-NN: {100 * ab:.1f}%   '
                                          f'median percentile known {np.median(pc[kind == 1]):.0f} / drifting {np.median(pc[kind == 2]):.0f}')

# ---------------------------------------------------------------- plot
PAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#8e5bd0', '#d6457a', '#5a8f29']
plt.rcParams.update({'font.size': 7, 'pdf.fonttype': 42})
fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.05), constrained_layout=True)
for ax, (name, _, A), Y, (_, _, ab) in zip(axes, PANELS, coords, out_rows):
    m = kind == 0; ax.scatter(Y[m, 0], Y[m, 1], s=1.5, c='#d0d0d0', lw=0, rasterized=True)
    for c in range(K):
        m = (kind == 1) & (fam == c); ax.scatter(Y[m, 0], Y[m, 1], s=2, c=PAL[c % 7], alpha=0.5, lw=0, rasterized=True)
    m = kind == 2; ax.scatter(Y[m, 0], Y[m, 1], s=4, c='black', marker='x', lw=0.4, alpha=0.8, rasterized=True)
    for c in range(K):
        m = (kind == 3) & (fam == c); ax.scatter(Y[m, 0], Y[m, 1], s=55, c=PAL[c % 7], marker='*', edgecolors='black', linewidths=0.5, zorder=5)
    ax.set_title(name, fontsize=7.5); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color('#bbbbbb')
    ax.text(0.03, 0.97, f'ROC-AUC {A:.2f}', transform=ax.transAxes, va='top', fontsize=6.5,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1))
from matplotlib.lines import Line2D
h = [Line2D([], [], marker='o', ls='none', ms=3, color='#888888', label='Known test (colour = family)'),
     Line2D([], [], marker='x', ls='none', ms=4, color='black', label=f'Drifting ({data["family"]})'),
     Line2D([], [], marker='o', ls='none', ms=3, color='#d0d0d0', label='Training'),
     Line2D([], [], marker='*', ls='none', ms=7, color='#888888', markeredgecolor='black', label='Centroid / prototype')]
fig.legend(handles=h, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, fontsize=6.5)
out = os.path.join(ROOT, 'figures'); os.makedirs(out, exist_ok=True)
tag = f'{a.ds.lower()}_{data["family"].lower()}'
fig.savefig(os.path.join(out, f'fig_tsne_{tag}.pdf'), bbox_inches='tight', dpi=300); plt.close(fig)
fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.05), constrained_layout=True)
cmap = plt.get_cmap('Purples')
for ax, (name, _, A), Y, pc in zip(axes, PANELS, coords, PCT):
    m = kind == 1
    sc = ax.scatter(Y[m, 0], Y[m, 1], s=2.5, c=pc[m], cmap=cmap, vmin=0, vmax=100, lw=0, rasterized=True)
    m = kind == 2
    ax.scatter(Y[m, 0], Y[m, 1], s=7, c=pc[m], cmap=cmap, vmin=0, vmax=100, marker='x', lw=0.6, rasterized=True)
    m = kind == 3
    ax.scatter(Y[m, 0], Y[m, 1], s=45, facecolors='white', marker='*', edgecolors='#d62728', linewidths=0.7, zorder=5)
    ax.set_title(name, fontsize=7.5); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color('#bbbbbb')
    ax.text(0.03, 0.97, f'ROC-AUC {A:.2f}', transform=ax.transAxes, va='top', fontsize=6.5,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1))
cb = fig.colorbar(sc, ax=axes, shrink=0.8, pad=0.01, aspect=25); cb.set_label('Drift score percentile', fontsize=6.5)
cb.ax.tick_params(labelsize=6)
h = [Line2D([], [], marker='o', ls='none', ms=3, color='#6a51a3', label='Known test'),
     Line2D([], [], marker='x', ls='none', ms=4, color='#6a51a3', label=f'Drifting ({data["family"]})'),
     Line2D([], [], marker='*', ls='none', ms=7, markerfacecolor='white', markeredgecolor='#d62728', color='white', label='Centroid / prototype')]
fig.legend(handles=h, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=6.5)
fig.savefig(os.path.join(out, f'fig_tsne_{tag}_score.pdf'), bbox_inches='tight', dpi=300); plt.close(fig)
with open(os.path.join(out, f'tsne_{tag}_coords.csv'), 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['panel', 'x', 'y', 'kind', 'family_idx', 'score_pct'])
    KN = ['train', 'known', 'drift', 'star']
    for (name, _, _), Y, pc in zip(PANELS, coords, PCT):
        for i in range(len(Y)):
            w.writerow([name, round(float(Y[i, 0]), 3), round(float(Y[i, 1]), 3), KN[kind[i]], int(fam[i]),
                        '' if np.isnan(pc[i]) else round(float(pc[i]), 1)])
print(f'Saved figures/fig_tsne_{tag}.pdf, figures/fig_tsne_{tag}_score.pdf and figures/tsne_{tag}_coords.csv')
