#!/usr/bin/env python3
"""Figures for the Drift Analysis section.

Reads (from --root, default: current directory)
  DX_methods.csv                              explanation methods per dataset x model
  explainability_ranking_fpr10_driftonly.csv  per-split explanation metrics (random-feature baseline)
  DX_reupload.csv                             distance to closest prototype after each re-upload
  DX_mixture.csv                              novelty residual and fidelity to each known prototype

Writes to --out (default: figures/drift), PDF and PNG:
  explanation_methods     sufficiency and CBP of each explanation method, per dataset and geometry
  reupload_decomposition  distance of known/drifting samples per layer, and their ratio
  family_mixture          fidelity of each withheld family to the known prototypes + novelty residual

Dataset and family names are read from the data; nothing dataset-specific is hardcoded.
Usage: python3 plot_drift_analysis.py [--root DIR] [--out DIR] [--figs reupload methods mixture]
       Default draws only reupload_decomposition, the one figure that needs the per-layer data.
"""
import argparse
import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# display labels for the result-directory names used in the ranking CSV; unknown names are shown as they are
DIR_LABELS = {
    'quantum_hybrid_v7_p702': 'Quantum (pair + prototype)',
    'quantum_hybrid_v7_p702_paironly': 'Quantum (pair loss only)',
    'classical_fidtwin_c_702': 'Twin, complex',
    'classical_fidtwin_p_702': 'Twin, plain',
    'classical_fidtwin_r_702': 'Twin, real',
    'cade_full_8_702': 'CADE-mini, full',
    'cade_enc_8_702': 'CADE-mini, encoder',
    'cade_tuned_8_702': 'CADE-mini, tuned',
    'cade_deep_8_702': 'CADE-mini, deep',
}
GROUP_ORDER = ['Quantum', 'Twins', 'CADE-minis']
GROUP_COLORS = {'Quantum': '#6a3d9a', 'Twins': '#1f78b4', 'CADE-minis': '#e6550d'}
METHOD_ORDER = ['random', 'boundary', 'coin', 'gradient', 'cade']
METHOD_LABELS = {'random': 'Random', 'boundary': 'Boundary', 'coin': 'COIN', 'gradient': 'Gradient', 'cade': 'CADE'}
METHOD_COLORS = {'random': '#bdbdbd', 'boundary': '#9ecae1', 'coin': '#74c476', 'gradient': '#fdae6b', 'cade': '#3f007d'}

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'legend.fontsize': 7,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'pdf.fonttype': 42, 'ps.fonttype': 42})


def group_of(label):
    l = label.lower()
    if l.startswith('quantum'):
        return 'Quantum'
    if l.startswith('twin'):
        return 'Twins'
    if l.startswith('cade'):
        return 'CADE-minis'
    return label


def canon_dataset(name, known):
    for k in known:
        if k.lower() == str(name).lower():
            return k
    return name


def save(fig, out, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out, f'{name}.{ext}'), bbox_inches='tight', dpi=300)
    plt.close(fig)
    print('wrote', os.path.join(out, name + '.{pdf,png}'))


def wmean(df, col, w='w'):
    d = df.dropna(subset=[col])
    return np.average(d[col], weights=d[w]) if len(d) and d[w].sum() > 0 else np.nan


# ------------------------------------------------------------------ explanation methods
def plot_methods(root, out):
    import pandas as pd  # only needed for the optional figures
    m = pd.read_csv(os.path.join(root, 'DX_methods.csv'))
    datasets = list(dict.fromkeys(m['dataset']))
    m['group'] = m['model'].map(group_of)
    m['w'] = m['n_drift_explained']

    # random-feature baseline: sufficiency of random features of CADE's size k, per split
    r = pd.read_csv(os.path.join(root, 'explainability_ranking_fpr10_driftonly.csv'))
    r['dataset'] = r['dataset'].map(lambda d: canon_dataset(d, datasets))
    r['label'] = r['model'].map(lambda x: DIR_LABELS.get(x, x))
    r['group'] = r['label'].map(group_of)
    r['w'] = r['n']

    groups = [g for g in GROUP_ORDER if g in set(m['group'])]
    fig, axes = plt.subplots(2, len(datasets), figsize=(7.0, 3.6), sharey='row', squeeze=False)
    width = 0.8 / len(METHOD_ORDER)
    x = np.arange(len(groups))
    for j, ds in enumerate(datasets):
        for i, met in enumerate(METHOD_ORDER):
            suff, cbp = [], []
            for g in groups:
                if met == 'random':
                    sub = r[(r['dataset'] == ds) & (r['group'] == g)]
                    suff.append(wmean(sub, 'suff_random'))
                    cbp.append(np.nan)
                else:
                    sub = m[(m['dataset'] == ds) & (m['group'] == g) & (m['method'] == met)]
                    suff.append(wmean(sub, 'sufficiency_fpr10'))
                    cbp.append(wmean(sub, 'cbp'))
            off = (i - (len(METHOD_ORDER) - 1) / 2) * width
            kw = dict(width=width, color=METHOD_COLORS[met], edgecolor='black', linewidth=0.3,
                      label=METHOD_LABELS[met])
            axes[0, j].bar(x + off, np.array(suff) * 100, **kw)
            if met != 'random':
                axes[1, j].bar(x + off, cbp, **kw)
        axes[0, j].set_title(ds)
        axes[1, j].axhline(0.5, ls='--', lw=0.7, color='black')
        for a in axes[:, j]:
            a.set_xticks(x)
            a.set_xticklabels(groups)
            a.grid(axis='y', lw=0.3, alpha=0.5)
            a.set_axisbelow(True)
    axes[0, 0].set_ylabel('Sufficiency (%)')
    axes[0, 0].set_ylim(0, 105)
    axes[1, 0].set_ylabel('Known-family percentile')
    axes[1, 0].set_ylim(0, 1.0)
    axes[1, 0].text(-0.45, 0.52, 'typical known sample', fontsize=6, va='bottom')
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=len(METHOD_ORDER), bbox_to_anchor=(0.5, 1.04), frameon=False)
    fig.tight_layout()
    save(fig, out, 'explanation_methods')


# ------------------------------------------------------------------ re-upload decomposition
def plot_reupload(root, out):
    # standard library only (no pandas): dataset, model, group, after_reupload, mean_distance
    data = defaultdict(dict)  # (dataset, model, group) -> {layer: distance}
    datasets, models_seen = [], []
    with open(os.path.join(root, 'DX_reupload.csv'), newline='') as fh:
        for r in csv.DictReader(fh):
            data[(r['dataset'], r['model'], r['group'])][int(r['after_reupload'])] = float(r['mean_distance'])
            if r['dataset'] not in datasets:
                datasets.append(r['dataset'])
            if r['model'] not in models_seen:
                models_seen.append(r['model'])
    # CADE objective first (solid), full objective second (dashed); other models follow in file order
    models = sorted(models_seen, key=lambda x: (0 if 'only' in x else 1, models_seen.index(x)))
    styles = ['-', '--', ':', '-.']
    gcol = {'known': '#7f7f7f', 'drifting': '#2A78D6'}  # paper palette: known grey, drifting = qcol
    fig, axes = plt.subplots(2, len(datasets), figsize=(7.0, 3.8), sharex=True, squeeze=False,
                             gridspec_kw={'height_ratios': [1.4, 1]})
    for j, ds in enumerate(datasets):
        for mi, mod in enumerate(models):
            ls = styles[mi % len(styles)]
            curves = {grp: data.get((ds, mod, grp), {}) for grp in ('known', 'drifting')}
            for grp, cur in curves.items():
                if cur:
                    xs = sorted(cur)
                    axes[0, j].plot(xs, [cur[x] for x in xs], ls=ls, color=gcol[grp], lw=1.2,
                                    label=f'{grp}, {mod}' if j == 0 else None)
            kn, dr = curves['known'], curves['drifting']
            xs = sorted(set(kn) & set(dr))
            if xs:
                axes[1, j].plot(xs, [dr[x] / kn[x] for x in xs], ls=ls, color='black', lw=1.2,
                                label=mod if j == 0 else None)
        axes[0, j].set_title(ds)
        axes[1, j].set_xlabel('Re-upload layer (0 = after encoding)')
        for a in axes[:, j]:
            a.grid(lw=0.3, alpha=0.5)
    axes[0, 0].set_ylabel('Distance to closest prototype')
    axes[1, 0].set_ylabel('Drifting / known')
    h0, l0 = axes[0, 0].get_legend_handles_labels()
    fig.legend(h0, l0, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.08), frameon=False)
    fig.tight_layout()
    save(fig, out, 'reupload_decomposition')


# ------------------------------------------------------------------ family mixture
def parse_fid(s):
    out = {}
    for part in str(s).split('|'):
        if ':' in part:
            k, v = part.rsplit(':', 1)
            out[k.strip()] = float(v)
    return out


def plot_mixture(root, out, model):
    import pandas as pd  # only needed for the optional figures
    d = pd.read_csv(os.path.join(root, 'DX_mixture.csv'))
    if model not in set(d['model']):
        model = d['model'].iloc[0]
        print('mixture model not found; using', model)
    d = d[d['model'] == model]
    datasets = list(dict.fromkeys(d['dataset']))
    fig = plt.figure(figsize=(7.0, 2.9 * len(datasets)))
    gs = fig.add_gridspec(len(datasets), 2, width_ratios=[2.2, 1], wspace=0.35, hspace=0.55)
    for i, ds in enumerate(datasets):
        sub = d[d['dataset'] == ds]
        withheld = list(dict.fromkeys(sub['unseen_family']))
        drift = sub[sub['group'] == 'drifting']
        known = sub[sub['group'] == 'known']
        protos = []
        rows, res_d, res_k = [], [], []
        for fam in withheld:
            dr = drift[drift['unseen_family'] == fam]
            kn = known[known['unseen_family'] == fam]
            fids = [parse_fid(s) for s in dr['mean_fidelity_to_prototypes']]
            keys = list(dict.fromkeys(k for f in fids for k in f))
            protos += [k for k in keys if k not in protos]
            rows.append({k: np.mean([f[k] for f in fids if k in f]) for k in keys})
            res_d.append(dr['novelty_residual'].mean())
            res_k.append(kn['novelty_residual'].mean() if len(kn) else np.nan)
        mat = np.array([[r.get(p, np.nan) for p in protos] for r in rows])

        ax = fig.add_subplot(gs[i, 0])
        im = ax.imshow(np.ma.masked_invalid(mat), cmap='Purples', vmin=0, vmax=np.nanmax(mat), aspect='auto')
        for a in range(mat.shape[0]):
            for b in range(mat.shape[1]):
                if not np.isnan(mat[a, b]):
                    ax.text(b, a, f'{mat[a, b]:.2f}', ha='center', va='center', fontsize=5.5,
                            color='white' if mat[a, b] > 0.6 * np.nanmax(mat) else 'black')
        ax.set_xticks(range(len(protos)))
        ax.set_xticklabels(protos, rotation=45, ha='right')
        ax.set_yticks(range(len(withheld)))
        ax.set_yticklabels(withheld)
        ax.set_xlabel('Known prototype')
        ax.set_ylabel('Withheld family')
        ax.set_title(f'{ds}: mean fidelity of drifting samples to each known prototype')
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

        ax2 = fig.add_subplot(gs[i, 1])
        y = np.arange(len(withheld))
        ax2.hlines(y, res_k, res_d, color='grey', lw=0.8)
        ax2.scatter(res_k, y, color='#1f78b4', s=14, label='known', zorder=3)
        ax2.scatter(res_d, y, color='#e31a1c', s=14, label='drifting', zorder=3)
        ax2.set_yticks(y)
        ax2.set_yticklabels(withheld)
        ax2.invert_yaxis()
        ax2.set_xlabel('Novelty residual')
        ax2.set_title(f'{ds}: novelty residual')
        ax2.grid(axis='x', lw=0.3, alpha=0.5)
        if i == 0:
            ax2.legend(loc='lower right', frameon=False)
    save(fig, out, 'family_mixture')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.')
    ap.add_argument('--out', default='figures/drift')
    ap.add_argument('--mixture-model', default='Quantum (pair + prototype)')
    ap.add_argument('--figs', nargs='+', default=['reupload'], choices=['reupload', 'methods', 'mixture'],
                    help="default: reupload only (the paper draws the methods and residual figures in pgfplots)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if 'methods' in a.figs:
        plot_methods(a.root, a.out)
    if 'reupload' in a.figs:
        plot_reupload(a.root, a.out)
    if 'mixture' in a.figs:
        plot_mixture(a.root, a.out, a.mixture_model)


if __name__ == '__main__':
    main()
