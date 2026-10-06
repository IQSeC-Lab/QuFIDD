#!/usr/bin/env python3
"""
plot_latent_spaces.py -- capacity_matched / capacity_matched_8fam (top level, next to evaluate.py)
======================================================================================================================
Latent-space maps of every trained model, from the saved results (nothing is retrained).

  quantum models and classical fidelity twins   states.npz (angle models: features.npz) -> 2-D kernel PCA of the
                                                FIDELITY kernel |<psi_i|psi_j>|^2, i.e. in the geometry they score
                                                with; two-register models use the product of the registers'
                                                fidelities (the fidelity of the joint product state)
  CADE-minis                                    latents RECOMPUTED from model_weights.pt (encoder rebuilt, applied to
                                                the same PCA input) -> 2-D PCA of the latent vectors (Euclidean)
  models without an embedding (hybrid v5's combiner, PCA-only baselines) are skipped

Each map shows a subsample of the training samples coloured by known family, and the held-out (drifting) test
samples in black. The 2-D map is fitted on the training subsample and the test samples are projected into it.

Outputs (per PCA folder), in <pca>/comparison/latent/:
  model_<folder>_run<r>.png     one figure per model: 8 panels, one per held-out family
  split<s>_<family>_run<r>.png  one figure per held-out family: all models side by side

Usage (from capacity_matched_8fam/):
  python3 plot_latent_spaces.py pca8 pca64                 # seed run 0, all models
  python3 plot_latent_spaces.py pca8 --run 1 --models hybrid_v7 fidtwin cade_enc
  python3 plot_latent_spaces.py pca8 --n-train 600 --n-test 1200
"""
import os, sys, glob, json, argparse, importlib.util, re
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA, PCA


def task_dir(root, folder, run, split):
    return os.path.join(root, folder, 'results', f'run{run}', f'split{split}')


def load_module(path):
    spec = importlib.util.spec_from_file_location(os.path.basename(path)[:-3], path); m = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, [path]; spec.loader.exec_module(m); sys.argv = argv
    return m


def subsample(n, k, seed):
    return np.sort(np.random.default_rng(seed).choice(n, min(n, k), replace=False))


def fidelity_embedding(d, n_tr, n_te, ood):
    """d: directory with states.npz / features.npz. Returns (Z_train, Z_test, y_train_sub, ood_sub, kind) or None."""
    f = os.path.join(d, 'states.npz') if os.path.exists(os.path.join(d, 'states.npz')) else os.path.join(d, 'features.npz')
    if not os.path.exists(f):
        return None
    z = np.load(f)
    if 'states_train' in z.files:
        regs_tr, regs_te = [z['states_train']], [z['states_test']]
    elif 'features_train' in z.files:
        regs_tr, regs_te = [z['features_train']], [z['features_test']]
    else:
        ks = sorted(k for k in z.files if re.match(r'states_train_r\d+$', k))
        if not ks:
            return None
        regs_tr = [z[k] for k in ks]; regs_te = [z[k.replace('train', 'test')] for k in ks]
    ytr = z['y_train']
    itr, ite = subsample(len(ytr), n_tr, 0), subsample(len(regs_te[0]), n_te, 1)
    def K(A, B):
        out = 1.0
        for a, b in zip(A, B):
            a = a.reshape(len(a), -1); b = b.reshape(len(b), -1)
            out = out * np.abs(a.conj() @ b.T) ** 2
        return out
    Atr = [r[itr] for r in regs_tr]; Ate = [r[ite] for r in regs_te]
    kp = KernelPCA(n_components=2, kernel='precomputed').fit(K(Atr, Atr))
    kind = 'kernel PCA of the fidelity kernel' + (' (product of %d registers)' % len(Atr) if len(Atr) > 1 else '')
    return kp.transform(K(Atr, Atr)), kp.transform(K(Ate, Atr)), ytr[itr], ood[ite], kind


def cade_embedding(root, folder, run, split, n_tr, n_te, ood, cache, data_dir=None):
    """Rebuild a CADE-mini encoder from model_weights.pt and recompute its latent vectors."""
    import torch
    d = task_dir(root, folder, run, split); w = os.path.join(d, 'model_weights.pt')
    script = glob.glob(os.path.join(root, folder, 'run_*.py'))
    if not os.path.exists(w) or not script:
        return None
    key = (script[0], split)
    if key not in cache:                                   # data + PCA are shared by every CADE-mini of this split
        mod = load_module(script[0])
        data = mod.load_split(os.path.expanduser(data_dir or mod.DEFAULT_DATA), mod.DEFAULT_PREFIX, split)
        pca = mod.fit_pca(data['Xtr'])
        cache.clear(); cache[key] = (pca.transform(data['Xtr']).astype(np.float32), pca.transform(data['Xte']).astype(np.float32), data['ytr'])
    Ztr, Zte, ytr = cache[key]
    sd = torch.load(w, map_location='cpu')
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    layers = sorted({int(k.split('.')[1]) for k in sd if k.startswith('0.') and k.endswith('.weight')})
    Ws = [(sd[f'0.{i}.weight'].numpy(), sd[f'0.{i}.bias'].numpy()) for i in layers]
    def enc(X):
        h = X
        for j, (Wm, b) in enumerate(Ws):
            h = h @ Wm.T + b
            if j < len(Ws) - 1:
                h = np.maximum(h, 0)                     # ReLU between layers, linear bottleneck
        return h
    itr, ite = subsample(len(ytr), n_tr, 0), subsample(len(Zte), n_te, 1)
    Ltr, Lte = enc(Ztr[itr]), enc(Zte[ite])
    p = PCA(n_components=2).fit(Ltr)
    return p.transform(Ltr), p.transform(Lte), ytr[itr], ood[ite], f'PCA of the {Ltr.shape[1]}-D latent (Euclidean)'


def discover(root, filters):
    out = []
    for f in sorted(os.listdir(root)):
        if not os.path.isdir(os.path.join(root, f, 'results')) or f.startswith('pca_only'):
            continue
        if filters and not any(x in f for x in filters):
            continue
        out.append(f)
    return out


def label_of(root, folder):
    s = glob.glob(os.path.join(root, folder, 'results', 'run*', 'split*', 'summary.json'))
    if s:
        m = json.load(open(s[0])).get('model', folder)
        return m if len(m) < 60 else m[:57] + '...'
    return folder


def draw(ax, emb, title):
    Ztr, Zte, ytr, ood, kind = emb
    ax.scatter(Ztr[:, 0], Ztr[:, 1], c=ytr, cmap='tab10', s=4, alpha=0.45, linewidths=0)
    o = ood.astype(bool)
    ax.scatter(Zte[o, 0], Zte[o, 1], c='k', s=5, alpha=0.6, linewidths=0, label='drift (test)')
    ax.set_title(title, fontsize=8.5); ax.set_xticks([]); ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('pca_dirs', nargs='+')
    ap.add_argument('--run', type=int, default=0); ap.add_argument('--models', nargs='*', default=None)
    ap.add_argument('--data-dir', default=None, help='override the data folder for the CADE latents');
    ap.add_argument('--n-train', type=int, default=800); ap.add_argument('--n-test', type=int, default=1500)
    a = ap.parse_args()
    for pd_ in a.pca_dirs:
        root = os.path.abspath(pd_); out = os.path.join(root, 'comparison', 'latent'); os.makedirs(out, exist_ok=True)
        folders = discover(root, a.models)
        splits = sorted({int(p.split('split')[-1]) for f in folders for p in glob.glob(os.path.join(root, f, 'results', f'run{a.run}', 'split*'))})
        print(f'===== {root}: {len(folders)} models, splits {splits}, seed run {a.run}', flush=True)
        E, fams, cache = {}, {}, {}
        for s in splits:
            for f in folders:
                d = task_dir(root, f, a.run, s)
                if not os.path.exists(os.path.join(d, 'ood_scores.csv')):
                    continue
                ood = np.genfromtxt(os.path.join(d, 'ood_scores.csv'), delimiter=',', names=True)['is_ood'].astype(int)
                fams[s] = json.load(open(os.path.join(d, 'summary.json'))).get('ood_family', f'split{s}')
                try:
                    emb = cade_embedding(root, f, a.run, s, a.n_train, a.n_test, ood, cache, a.data_dir) if f.startswith('cade') \
                        else fidelity_embedding(d, a.n_train, a.n_test, ood)
                except Exception as ex:
                    print(f'  {f} split {s}: skipped ({ex})', flush=True); emb = None
                if emb is not None:
                    E[(f, s)] = emb
            print(f'  split {s} ({fams.get(s)}): {sum(1 for k in E if k[1] == s)} models embedded', flush=True)
        models = [f for f in folders if any((f, s) in E for s in splits)]
        skipped = [f for f in folders if f not in models]
        if skipped:
            print(f'  no embedding (skipped): {skipped}')
        # one figure per model: a panel per held-out family
        for f in models:
            n = len(splits); cols = 4; rows = (n + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.9 * rows), squeeze=False)
            for ax, s in zip(axes.ravel(), splits):
                if (f, s) in E: draw(ax, E[(f, s)], f'{fams[s]} held out')
                else: ax.axis('off')
            for ax in axes.ravel()[n:]: ax.axis('off')
            kind = next(E[(f, s)][4] for s in splits if (f, s) in E)
            fig.suptitle(f'{label_of(root, f)} — {kind}; seed run {a.run}\ncoloured: known families (train) · black: drifting family (test)', fontsize=10)
            fig.tight_layout(rect=[0, 0, 1, 1 - 0.75 / (3.9 * rows + 0.75)]); fig.savefig(os.path.join(out, f'model_{f}_run{a.run}.png'), dpi=130); plt.close(fig)
        # one figure per held-out family: all models
        for s in splits:
            ms = [f for f in models if (f, s) in E]
            cols = 4; rows = (len(ms) + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.9 * rows), squeeze=False)
            for ax, f in zip(axes.ravel(), ms):
                draw(ax, E[(f, s)], f'{f}\n({E[(f, s)][4]})')
            for ax in axes.ravel()[len(ms):]: ax.axis('off')
            fig.suptitle(f'{os.path.basename(root)}: {fams[s]} held out (split {s}), seed run {a.run}\ncoloured: known families (train) · black: drifting family (test)', fontsize=11)
            fig.tight_layout(rect=[0, 0, 1, 1 - 0.85 / (3.9 * rows + 0.85)]); fig.savefig(os.path.join(out, f'split{s}_{fams[s]}_run{a.run}.png'), dpi=120); plt.close(fig)
        print(f'  saved {len(models)} model figures and {len(splits)} family figures -> {out}', flush=True)


if __name__ == '__main__':
    main()
