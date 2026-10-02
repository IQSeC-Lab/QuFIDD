#!/usr/bin/env python3
"""
analyze_family.py -- Full_new_experiments (top level)   (standalone script)
===========================================================================
Why does a model place a held-out family close to the known families? Diagnostic
plots for one held-out family (default: plankton) across the PCA-5 models, from
files the experiments already saved -- nothing is retrained.

Models (each skipped if its results are missing):
  PCA-5 L2 (input geometry) | CADE-mini selected | CADE-mini full | Quantum v1 | Quantum v2 | Quantum v3
Distances are the ones each model is scored with: Euclidean to class centroids (classical),
1 - <psi|rho_c|psi> to class density-matrix prototypes (quantum).

Per task (3 seeds of the family) -> analysis_<family>/run{r}_split{s}/:
  1_latent_spaces.png      each model's space in 2-D (PCA for classical; kernel PCA with the
                           fidelity kernel for quantum): known training families + held-out test samples
  2_nearest_family.png     which known family each held-out sample is closest to (vs how known
                           test samples are assigned) -> which family the held-out one is confused with
  3_distance_overlap.png   distance to the nearest prototype: known test vs held-out test, with the
                           separation expressed as ROC-AUC -> how much the distributions overlap
  4_mad_scores.png         MAD anomaly score of held-out samples per nearest family, vs T = 3.5
  5_input_similarity.png   in the PCA-5 INPUT (before any model): cosine similarity of the held-out
                           family to each known family, vs the known families' own cohesion
  summary.txt              the numbers behind every plot
Across the family's seeds -> analysis_<family>/summary_all_seeds.txt

Usage (from Full_new_experiments/):
  python3 analyze_family.py                       # plankton, all 3 seeds
  python3 analyze_family.py --family droidkungfu  # any held-out family, for contrast
  python3 analyze_family.py --tsne                # add a t-SNE row to the latent figure (slower)
"""
import os, json, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.abspath(__file__))
SEEDS = {0: 42, 1: 123, 2: 456}
N_SPLITS, PCA_DIM, MAD_T, MAD_B = 4, 5, 3.5, 1.4826


def load_split(data_dir, prefix, s):
    d = np.load(os.path.join(os.path.expanduser(data_dir), f'{prefix}_{s}.npz'), allow_pickle=True)
    dense = lambda a: a.item().toarray().astype(np.float32) if a.shape == () else a.astype(np.float32)
    hf = d['heldout_family']; fam = str(hf.item()) if hf.shape == () else str(hf)
    return dense(d['X_train']), d['y_train'].astype(np.int64), dense(d['X_test']), d['y_test'].astype(np.int64), fam


def family_names(data_dir, prefix):
    out = {}
    for s in range(N_SPLITS):
        h = np.load(os.path.join(os.path.expanduser(data_dir), f'{prefix}_{s}.npz'), allow_pickle=True)['heldout_family']
        out[s] = str(h.item()) if h.shape == () else str(h)
    return out


def encoder_from(path, hidden, with_decoder, n_lat):
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
    ml = nn.ModuleList(mods); ml.load_state_dict(torch.load(path, map_location='cpu'))
    return lambda Z: ml[0](torch.from_numpy(np.asarray(Z, np.float32))).detach().numpy()


def representations(r, s, Ztr, Zte, n_known):
    """Returns {model: (kind, train_repr, test_repr)} for every model whose files exist."""
    reps = {}
    l2 = lambda Z: Z / np.linalg.norm(Z, axis=1, keepdims=True).clip(min=1e-8)
    reps['PCA-5 L2 (input)'] = ('euc', l2(Ztr), l2(Zte))
    t = lambda folder: os.path.join(ROOT, folder, 'results', f'run{r}', f'split{s}')
    sel = os.path.join(ROOT, 'cade_mini_select', 'selection', 'selected_variant.json')
    p = os.path.join(t('cade_mini_select'), 'model_weights.pt')
    if os.path.exists(p) and os.path.exists(sel):
        design = json.load(open(sel))['selected']['variant']['design']
        hid, dec = ([22], False) if design == 'enc' else ([11], True)
        f = encoder_from(p, hid, dec, n_known); reps['CADE-mini selected'] = ('euc', f(Ztr), f(Zte))
    p = os.path.join(t('cade_mini_full'), 'model_weights.pt')
    if os.path.exists(p):
        f = encoder_from(p, [11], True, n_known); reps['CADE-mini full'] = ('euc', f(Ztr), f(Zte))
    for name, folder, fn, ktr, kte in [('Quantum v1', 'quantum', 'states.npz', 'states_train', 'states_test'),
                                      ('Quantum v2', 'quantum_v2', 'features.npz', 'features_train', 'features_test'),
                                      ('Quantum v3', 'quantum_v3', 'states.npz', 'states_train', 'states_test')]:
        p = os.path.join(t(folder), fn)
        if os.path.exists(p):
            z = np.load(p)
            if np.iscomplexobj(z[ktr]):
                reps[name] = ('q', z[ktr], z[kte])
    return reps


def class_distances(kind, Rtr, ytr, R):
    """[n, C] distance of every sample in R to every known class, as the model scores it."""
    C = len(np.unique(ytr))
    if kind == 'euc':
        return np.stack([np.linalg.norm(R - Rtr[ytr == c].mean(0), axis=1) for c in range(C)], 1)
    return np.stack([1 - (np.abs(R.conj() @ Rtr[ytr == c].T) ** 2).mean(1) for c in range(C)], 1)


def own_train_distances(kind, Rtr, ytr):
    C = len(np.unique(ytr)); own = []
    for c in range(C):
        m = ytr == c
        if kind == 'euc':
            own.append(np.linalg.norm(Rtr[m] - Rtr[m].mean(0), axis=1))
        else:
            F = np.abs(Rtr[m].conj() @ Rtr[m].T) ** 2; n = m.sum()
            own.append(1 - (F.sum(1) - 1) / (n - 1))
    return own


def project2d(kind, Rtr, Rte, sub):
    if kind == 'euc':
        p = PCA(n_components=2).fit(Rtr); return p.transform(Rtr[sub]), p.transform(Rte), 'PCA'
    K = np.abs(Rtr[sub].conj() @ Rtr[sub].T) ** 2
    kp = KernelPCA(n_components=2, kernel='precomputed').fit(K)
    return kp.transform(K), kp.transform(np.abs(Rte.conj() @ Rtr[sub].T) ** 2), 'kernel PCA (fidelity)'


def analyze_task(r, s, data_dir, prefix, tsne, outroot):
    Xtr, ytr_raw, Xte, yte, fam = load_split(data_dir, prefix, s)
    names = family_names(data_dir, prefix)
    known = sorted(int(c) for c in np.unique(ytr_raw)); ytr = np.array([known.index(int(c)) for c in ytr_raw])
    ood = ~np.isin(yte, known); kn_te = ~ood
    yte_idx = np.array([known.index(int(c)) if c in known else -1 for c in yte])
    kname = [names.get(c, str(c)) for c in known]
    pca = PCA(n_components=PCA_DIM, svd_solver='full').fit(Xtr)
    Ztr, Zte = pca.transform(Xtr), pca.transform(Xte)
    reps = representations(r, s, Ztr, Zte, len(known))
    out = os.path.join(outroot, f'run{r}_split{s}'); os.makedirs(out, exist_ok=True)
    cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
    lines, stats = [], {}
    def log(x=''):
        print(x, flush=True); lines.append(x)
    log(f'=== {fam} held out | run {r} (seed {SEEDS[r]}) split {s} | known: {kname} | models: {list(reps)}')

    D, A = {}, {}
    for m, (kind, Rtr, Rte) in reps.items():
        Dm = class_distances(kind, Rtr, ytr, Rte); own = own_train_distances(kind, Rtr, ytr)
        med = np.array([np.median(o) for o in own]); mad = np.array([max(MAD_B * np.median(np.abs(o - np.median(o))), 1e-12) for o in own])
        D[m], A[m] = Dm, np.abs(Dm - med) / mad
        raw = Dm.min(1); nearest = Dm.argmin(1)
        near_frac = [float((nearest[ood] == c).mean()) for c in range(len(known))]
        acc_known = float((nearest[kn_te] == yte_idx[kn_te]).mean())
        ratio = float(np.median(raw[ood]) / np.median(raw[kn_te]))
        spread = float(np.mean([np.median(o) for o in own]))
        stats[m] = {'ROC_raw': float(roc_auc_score(ood, raw)), 'ROC_mad': float(roc_auc_score(ood, A[m].min(1))),
                    'nearest_frac': near_frac, 'known_assign_acc': acc_known,
                    'median_ratio_ood_over_known': ratio, 'own_spread': spread,
                    'flagged_ood@3.5': float((A[m].min(1)[ood] > MAD_T).mean()),
                    'flagged_known@3.5': float((A[m].min(1)[kn_te] > MAD_T).mean())}
        log(f'  {m:<20} ROC_raw {stats[m]["ROC_raw"]:.3f}  ROC_mad {stats[m]["ROC_mad"]:.3f}  '
            f'median dist {fam}/known {ratio:.2f}  nearest family of {fam}: '
            + ', '.join(f'{kname[c]} {near_frac[c]:.0%}' for c in range(len(known)))
            + f'  | flagged@3.5: {fam} {stats[m]["flagged_ood@3.5"]:.0%}, known {stats[m]["flagged_known@3.5"]:.0%}')

    # 1. latent spaces
    sub = np.random.default_rng(0).choice(len(ytr), min(800, len(ytr)), replace=False)
    rows = 2 if tsne else 1
    fig, ax = plt.subplots(rows, len(reps), figsize=(4.3 * len(reps), 4.3 * rows), squeeze=False)
    for j, (m, (kind, Rtr, Rte)) in enumerate(reps.items()):
        Ptr, Pte, lab = project2d(kind, Rtr, Rte, sub)
        a = ax[0, j]
        for c in range(len(known)):
            mk = ytr[sub] == c; a.scatter(Ptr[mk, 0], Ptr[mk, 1], s=5, alpha=0.3, color=cols[c], label=kname[c])
        a.scatter(Pte[ood, 0], Pte[ood, 1], s=30, marker='*', color='k', edgecolors='w', linewidths=0.3, alpha=0.8, label=f'{fam} (held out)')
        a.set_title(f'{m}\n{lab} | ROC_raw {stats[m]["ROC_raw"]:.2f}', fontsize=9); a.grid(alpha=0.3)
        if j == 0:
            a.legend(fontsize=7, markerscale=2)
        if tsne:
            from sklearn.manifold import TSNE
            if kind == 'euc':
                T = TSNE(2, init='pca', random_state=0, perplexity=30).fit_transform(np.vstack([Rtr[sub], Rte]))
            else:
                S = np.vstack([Rtr[sub], Rte]); Dq = np.sqrt(np.clip(1 - np.abs(S.conj() @ S.T) ** 2, 0, 1)); np.fill_diagonal(Dq, 0)
                T = TSNE(2, metric='precomputed', init='random', random_state=0, perplexity=30).fit_transform(Dq)
            b = ax[1, j]; n = len(sub)
            for c in range(len(known)):
                mk = ytr[sub] == c; b.scatter(T[:n][mk, 0], T[:n][mk, 1], s=5, alpha=0.3, color=cols[c])
            b.scatter(T[n:][ood, 0], T[n:][ood, 1], s=30, marker='*', color='k', edgecolors='w', linewidths=0.3, alpha=0.8)
            b.set_title(f'{m} | t-SNE', fontsize=9); b.grid(alpha=0.3)
    fig.suptitle(f'Where does the held-out family ({fam}) land? run {r} (seed {SEEDS[r]})', fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(out, '1_latent_spaces.png'), dpi=130); plt.close(fig)

    # 2. nearest family
    fig, ax = plt.subplots(figsize=(1.9 * len(reps) + 3, 4.2)); w = 0.8 / len(known); x0 = np.arange(len(reps))
    for c in range(len(known)):
        ax.bar(x0 + (c - (len(known) - 1) / 2) * w, [stats[m]['nearest_frac'][c] for m in reps], w, color=cols[c], label=f'nearest = {kname[c]}')
    ax.set_xticks(x0); ax.set_xticklabels(list(reps), rotation=20, fontsize=8); ax.set_ylim(0, 1)
    ax.set_ylabel(f'share of {fam} test samples'); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')
    ax.set_title(f'Which known family is each {fam} sample closest to?  (known-sample assignment accuracy: '
                 + ', '.join(f'{m.split(" (")[0]} {stats[m]["known_assign_acc"]:.0%}' for m in reps) + ')', fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, '2_nearest_family.png'), dpi=130); plt.close(fig)

    # 3. distance overlap
    fig, ax = plt.subplots(1, len(reps), figsize=(4.0 * len(reps), 3.6), squeeze=False)
    for a, m in zip(ax[0], reps):
        raw = D[m].min(1); bins = np.linspace(np.percentile(raw, 0.5), np.percentile(raw, 99.5), 45)
        a.hist(raw[kn_te], bins=bins, density=True, alpha=0.55, label='known test')
        a.hist(raw[ood], bins=bins, density=True, alpha=0.55, color='k', label=f'{fam} test')
        a.set_title(f'{m}\nROC_raw {stats[m]["ROC_raw"]:.2f} | median ratio {stats[m]["median_ratio_ood_over_known"]:.2f}', fontsize=8)
        a.set_xlabel('distance to nearest known prototype', fontsize=8); a.grid(alpha=0.3)
    ax[0, 0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, '3_distance_overlap.png'), dpi=130); plt.close(fig)

    # 4. MAD scores of held-out samples by nearest family
    fig, ax = plt.subplots(1, len(reps), figsize=(4.0 * len(reps), 3.6), squeeze=False)
    for a, m in zip(ax[0], reps):
        near = D[m].argmin(1); Amin = A[m].min(1)
        data_ = [Amin[ood & (near == c)] for c in range(len(known))] + [Amin[kn_te]]
        a.boxplot([d if len(d) else [np.nan] for d in data_], showfliers=False)
        a.set_xticks(range(1, len(known) + 2)); a.set_xticklabels([f'{fam}\nnear {k}' for k in kname] + ['known\ntest'], fontsize=6)
        a.axhline(MAD_T, color='r', ls='--', lw=1); a.set_yscale('symlog'); a.grid(alpha=0.3)
        a.set_title(f'{m}\nflagged@3.5: {fam} {stats[m]["flagged_ood@3.5"]:.0%}, known {stats[m]["flagged_known@3.5"]:.0%}', fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, '4_mad_scores.png'), dpi=130); plt.close(fig)

    # 5. input similarity (PCA-5, before any model)
    U = lambda Z: Z / np.linalg.norm(Z, axis=1, keepdims=True).clip(min=1e-8)
    Utr, Ute = U(Ztr), U(Zte)
    cent = np.stack([U(Utr[ytr == c].mean(0, keepdims=True))[0] for c in range(len(known))])
    cos_ood = Ute[ood] @ cent.T
    cos_own = [Utr[ytr == c] @ cent[c] for c in range(len(known))]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot([cos_own[c] for c in range(len(known))] + [cos_ood[:, c] for c in range(len(known))], showfliers=False)
    ax.set_xticks(range(1, 2 * len(known) + 1))
    ax.set_xticklabels([f'{k}\nto own centroid' for k in kname] + [f'{fam}\nto {k}' for k in kname], fontsize=7)
    ax.axvline(len(known) + 0.5, color='k', lw=0.8); ax.set_ylabel('cosine similarity (PCA-5 direction)'); ax.grid(alpha=0.3)
    ax.set_title(f'Input geometry: is {fam} already close to a known family before any model?', fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(out, '5_input_similarity.png'), dpi=130); plt.close(fig)
    log(f'  INPUT (PCA-5 cosine): {fam} vs known centroids median ' + ', '.join(f'{kname[c]} {np.median(cos_ood[:, c]):.2f}' for c in range(len(known)))
        + ' | known families to own centroid median ' + ', '.join(f'{kname[c]} {np.median(cos_own[c]):.2f}' for c in range(len(known))))
    open(os.path.join(out, 'summary.txt'), 'w').write('\n'.join(lines) + '\n')
    return fam, kname, stats


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--family', default='plankton')
    ap.add_argument('--data-dir', default='~/projects/OOD/Data/cade_lamda_4class_vt')
    ap.add_argument('--prefix', default='drebin_new')
    ap.add_argument('--tsne', action='store_true')
    a = ap.parse_args()
    names = family_names(a.data_dir, a.prefix)
    splits = [s for s, f in names.items() if f.lower() == a.family.lower()]
    if not splits:
        raise SystemExit(f'No split holds out {a.family}; families: {list(names.values())}')
    s = splits[0]; outroot = os.path.join(ROOT, f'analysis_{a.family}')
    allstats = []
    for r in range(3):
        allstats.append(analyze_task(r, s, a.data_dir, a.prefix, a.tsne, outroot))
    fam, kname, _ = allstats[0]; models = list(allstats[0][2])
    L = [f'=== {fam} held out: mean over {len(allstats)} seeds ===']
    for m in models:
        st = [x[2][m] for x in allstats if m in x[2]]
        nf = np.mean([x['nearest_frac'] for x in st], 0)
        L.append(f'  {m:<20} ROC_raw {np.mean([x["ROC_raw"] for x in st]):.3f}  ROC_mad {np.mean([x["ROC_mad"] for x in st]):.3f}  '
                 f'median dist ratio {np.mean([x["median_ratio_ood_over_known"] for x in st]):.2f}  nearest: '
                 + ', '.join(f'{kname[c]} {nf[c]:.0%}' for c in range(len(kname)))
                 + f'  | flagged@3.5 {fam} {np.mean([x["flagged_ood@3.5"] for x in st]):.0%}, known {np.mean([x["flagged_known@3.5"] for x in st]):.0%}')
    print('\n'.join(L)); open(os.path.join(outroot, 'summary_all_seeds.txt'), 'w').write('\n'.join(L) + '\n')
    print(f'\nSaved plots and summaries under {outroot}/')
