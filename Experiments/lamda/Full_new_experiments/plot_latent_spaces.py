#!/usr/bin/env python3
"""
plot_latent_spaces.py -- Full_new_experiments (top level)   (standalone script)
==============================================================================
Latent-space figures in the 2 x 2 style:

      t-SNE | Train          t-SNE | Test  (known families + OOD as stars)
      PCA   | Train          PCA   | Test

for every model and task, from files the experiments already saved (no retraining):
  pca_only                the PCA-5 coordinates themselves
  cade_mini_full          3-D latent of the saved encoder (5-11-3)
  cade_mini_encoder_only  3-D latent of the saved encoder (5-22-3)
  quantum                 the saved 32-dim quantum states (states.npz)

Train and test are placed in the SAME coordinates, so the panels are comparable:
  - PCA is fit on the training latents and applied to the test latents.
    Quantum: kernel PCA with the fidelity kernel |<psi|psi'>|^2 (phase-invariant, the
    geometry the model actually uses), fit on training states, applied to test states.
  - t-SNE is run once on train + test together, then split into the two panels.
    Quantum: t-SNE on the distance sqrt(1 - fidelity).
  Both panels of a row share axis limits.

Usage (from Full_new_experiments/):
  python3 plot_latent_spaces.py                     # task 0, all four models
  python3 plot_latent_spaces.py 5                   # any task 0-11
  python3 plot_latent_spaces.py all                 # all 12 tasks
  python3 plot_latent_spaces.py 0 --models quantum cade_mini_full
  python3 plot_latent_spaces.py 0 --max-train 1200  # subsample training points for t-SNE speed
Output: <model>/latent_plots/latent_space_run{r}_split{s}_{family}.png
"""
import os, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE

ROOT = os.path.dirname(os.path.abspath(__file__))
SEEDS = {0: 42, 1: 123, 2: 456}
N_SPLITS, PCA_DIM = 4, 5
DISPLAY = {'pca_only': 'PCA-5 only', 'cade_mini_full': 'CADE-mini full',
           'cade_mini_encoder_only': 'CADE-mini encoder-only', 'quantum': 'Quantum (re-uploading)'}
ENCODERS = {'cade_mini_full': ([11], True), 'cade_mini_encoder_only': ([22], False)}


def load_split(data_dir, prefix, s):
    d = np.load(os.path.join(os.path.expanduser(data_dir), f'{prefix}_{s}.npz'), allow_pickle=True)
    dense = lambda a: a.item().toarray().astype(np.float32) if a.shape == () else a.astype(np.float32)
    return dense(d['X_train']), d['y_train'].astype(np.int64), dense(d['X_test']), d['y_test'].astype(np.int64)


def family_names(data_dir, prefix):
    """label i = held-out family of split i (verified: OOD label = split index)."""
    names = {}
    for s in range(N_SPLITS):
        hf = np.load(os.path.join(os.path.expanduser(data_dir), f'{prefix}_{s}.npz'), allow_pickle=True)['heldout_family']
        names[s] = str(hf.item()) if hf.shape == () else str(hf)
    return names


def load_encoder(path, hidden, with_decoder, n_lat):
    import torch, torch.nn as nn
    dims = [PCA_DIM] + hidden + [n_lat]
    enc = []
    for i in range(len(dims) - 2):
        enc += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
    enc += [nn.Linear(dims[-2], dims[-1])]
    mods = [nn.Sequential(*enc)]
    if with_decoder:
        dec = []
        for i in range(len(dims) - 1, 1, -1):
            dec += [nn.Linear(dims[i], dims[i - 1]), nn.ReLU()]
        dec += [nn.Linear(dims[1], dims[0])]
        mods.append(nn.Sequential(*dec))
    ml = nn.ModuleList(mods)
    ml.load_state_dict(torch.load(path, map_location='cpu'))
    return lambda Z: ml[0](torch.from_numpy(np.asarray(Z, np.float32))).detach().numpy()


def project(model, Ltr, Lte, seed):
    """Returns (pca_tr, pca_te, tsne_tr, tsne_te, pca_label)."""
    n = len(Ltr)
    if model == 'quantum':
        Ktr = np.abs(Ltr.conj() @ Ltr.T) ** 2
        kp = KernelPCA(n_components=2, kernel='precomputed').fit(Ktr)
        p_tr, p_te = kp.transform(Ktr), kp.transform(np.abs(Lte.conj() @ Ltr.T) ** 2)
        S = np.vstack([Ltr, Lte])
        D = np.sqrt(np.clip(1 - np.abs(S.conj() @ S.T) ** 2, 0, 1)); np.fill_diagonal(D, 0)
        T = TSNE(n_components=2, metric='precomputed', init='random', random_state=seed, perplexity=30).fit_transform(D)
        return p_tr, p_te, T[:n], T[n:], 'Kernel PCA (fidelity kernel)'
    p = PCA(n_components=2).fit(Ltr)
    L = np.vstack([Ltr, Lte])
    T = TSNE(n_components=2, init='pca', random_state=seed, perplexity=30).fit_transform(L)
    return p.transform(Ltr), p.transform(Lte), T[:n], T[n:], 'PCA'


def plot(model, r, s, fam_of, ytr, yte, ood_label, P_tr, P_te, T_tr, T_te, plabel, out):
    cmap = plt.get_cmap('tab10'); col = {lab: cmap(i % 10) for i, lab in enumerate(sorted(fam_of))}
    fig, ax = plt.subplots(2, 2, figsize=(14, 12))
    rows = [('t-SNE', T_tr, T_te), (plabel, P_tr, P_te)]
    for i, (name, A, B) in enumerate(rows):
        both = np.vstack([A, B]); pad = 0.05 * (both.max(0) - both.min(0))
        lim = (both.min(0) - pad, both.max(0) + pad)
        a = ax[i, 0]
        for lab in sorted(np.unique(ytr)):
            m = ytr == lab
            a.scatter(A[m, 0], A[m, 1], s=14, alpha=0.6, color=col[lab], edgecolors='none', label=fam_of[lab])
        a.set_title(f'{name} | Train'); a.legend(fontsize=8, loc='best', markerscale=1.5)
        a = ax[i, 1]
        for lab in sorted(np.unique(yte)):
            m = yte == lab
            if lab == ood_label:
                continue
            a.scatter(B[m, 0], B[m, 1], s=14, alpha=0.5, color=col[lab], edgecolors='none', label=fam_of[lab])
        m = yte == ood_label
        a.scatter(B[m, 0], B[m, 1], s=110, marker='*', color='steelblue', edgecolors='black', linewidths=0.7,
                  label=f'{fam_of[ood_label]} (OOD)', zorder=5)
        a.set_title(f'{name} | Test'); a.legend(fontsize=8, loc='best', markerscale=1.0)
        for a in ax[i]:
            a.set_xlim(lim[0][0], lim[1][0]); a.set_ylim(lim[0][1], lim[1][1])
            a.set_xlabel('Component 1'); a.set_ylabel('Component 2'); a.grid(alpha=0.3)
    fig.suptitle(f'{DISPLAY[model]} Latent Space -- Run {r} (seed {SEEDS[r]}) | OOD: {fam_of[ood_label]}',
                 fontsize=15, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(out, f'latent_space_run{r}_split{s}_{fam_of[ood_label]}.png')
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def run(task_id, models, data_dir, prefix, max_train):
    r, s = task_id // N_SPLITS, task_id % N_SPLITS
    seed = SEEDS[r]
    Xtr, ytr, Xte, yte = load_split(data_dir, prefix, s)
    fam_of = family_names(data_dir, prefix)
    known = sorted(int(c) for c in np.unique(ytr)); ood_label = [c for c in np.unique(yte) if c not in known][0]
    pca = PCA(n_components=PCA_DIM, svd_solver='full').fit(Xtr)          # identical to the run scripts
    Ztr, Zte = pca.transform(Xtr), pca.transform(Xte)
    sub = np.random.default_rng(0).choice(len(ytr), min(max_train, len(ytr)), replace=False) if max_train else np.arange(len(ytr))
    for model in models:
        tdir = os.path.join(ROOT, model, 'results', f'run{r}', f'split{s}')
        if model == 'pca_only':
            Ltr, Lte = Ztr, Zte
        elif model == 'quantum':
            f = os.path.join(tdir, 'states.npz')
            if not os.path.exists(f):
                print(f'  skip {model}: {f} not found'); continue
            z = np.load(f); Ltr, Lte = z['states_train'], z['states_test']
        else:
            f = os.path.join(tdir, 'model_weights.pt')
            if not os.path.exists(f):
                print(f'  skip {model}: {f} not found'); continue
            hid, dec = ENCODERS[model]
            enc = load_encoder(f, hid, dec, len(known)); Ltr, Lte = enc(Ztr), enc(Zte)
        P_tr, P_te, T_tr, T_te, plabel = project(model, Ltr[sub], Lte, seed)
        out = os.path.join(ROOT, model, 'latent_plots'); os.makedirs(out, exist_ok=True)
        path = plot(model, r, s, fam_of, ytr[sub], yte, ood_label, P_tr, P_te, T_tr, T_te, plabel, out)
        print(f'  task {task_id} {model:<24} -> {path}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('task_id', nargs='?', default='0', help="0-11 or 'all' (default 0)")
    p.add_argument('--models', nargs='+', default=list(DISPLAY), choices=list(DISPLAY))
    p.add_argument('--data-dir', default='~/projects/OOD/Data/cade_lamda_4class_vt')
    p.add_argument('--prefix', default='drebin_new')
    p.add_argument('--max-train', type=int, default=0, help='subsample training points (0 = all)')
    a = p.parse_args()
    for t in (range(3 * N_SPLITS) if a.task_id == 'all' else [int(a.task_id)]):
        run(t, a.models, a.data_dir, a.prefix, a.max_train)
