#!/usr/bin/env python3
"""plot_inspection.py -- CADE-style inspection curves (precision and recall vs. inspected samples) from the saved
ood_scores.csv files. Run from Experiments2 (via run_plot_inspection.slurm):
    python3 plot_inspection.py [--family EMBER=lethic --family Drebin=fakedoc --family lamda=basebridge] [--run 0]
Outputs in figures/:
  fig_inspection_examples_proto.pdf  one setting per dataset (CADE Fig. 4 style), seed run --run
  fig_inspection_mean_proto.pdf      mean over all 24 tasks, x = inspection effort (inspected / #drifting), 0..2
  inspection_stats_proto.csv         per model and dataset: mean effort, precision and recall at the ranked-F1 stopping point"""
import os, glob, json, argparse, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
DS = ['EMBER', 'Drebin', 'lamda']
DS_NAME = {'EMBER': 'EMBER', 'Drebin': 'Drebin', 'lamda': 'LAMDA'}
# (label, folder relative to <ds>/pca8, colour, line width); best twin / best CADE-mini by mean ROC-AUC per dataset
QC = ('Quantum (pair)', 'ablation_objective/quantum_hybrid_v7_p702_paironly', '#1b3a6b', 2.0)
QF = ('Quantum (pair + prototype)', 'quantum_pca8/pennylane/quantum_hybrid_v7_p702', '#3f7fd0', 2.0)
MODELS = {
    'EMBER': [QC, QF, ('Twin, complex (pair + prototype)', 'twin_pca8_p702/classical_fidtwin_c_702', '#e08a1e', 1.5),
              ('CADE-mini, encoder (pair + prototype)', 'cade_pca8_p702_proto/cade_enc_8_702_proto/tau_1', '#8c8c8c', 1.5)],
    'Drebin': [QC, QF, ('Twin, complex (pair + prototype)', 'twin_pca8_p702/classical_fidtwin_c_702', '#e08a1e', 1.5),
               ('CADE-mini, tuned (pair + prototype)', 'cade_pca8_p702_proto/cade_tuned_8_702_proto/tau_10', '#8c8c8c', 1.5)],
    'lamda': [QC, QF, ('Twin, plain (pair + prototype)', 'twin_pca8_p702/classical_fidtwin_p_702', '#e08a1e', 1.5),
              ('CADE-mini, deep (pair + prototype)', 'cade_pca8_p702_proto/cade_deep_8_702_proto/tau_1', '#8c8c8c', 1.5)],
}


ROLE = lambda lab: ('Best quantum-inspired twin' if lab.startswith('Twin') else 'Best CADE-mini' if lab.startswith('CADE') else lab)


def load_task(ds, folder, run, split):
    d = os.path.join(ROOT, ds, 'pca8', folder, 'results', f'run{run}', f'split{split}')
    f = os.path.join(d, 'ood_scores.csv')
    if not os.path.exists(f):
        return None
    with open(f) as fh:
        rd = csv.DictReader(fh); rows = [(float(r['raw_score']), int(float(r['is_ood']))) for r in rd]
    s = np.array([r[0] for r in rows]); y = np.array([r[1] for r in rows])
    fam = json.load(open(os.path.join(d, 'summary.json'))).get('ood_family', '?')
    return s, y, str(fam)


def curves(s, y):
    o = np.argsort(-s, kind='stable'); y = y[o]
    k = np.arange(1, len(y) + 1); tp = np.cumsum(y)
    prec = tp / k; rec = tp / max(y.sum(), 1)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0)
    b = int(np.argmax(f1))
    return k, prec, rec, b


def find_split(ds, family, run):
    for sp in range(8):
        t = load_task(ds, QF[1], run, sp) or load_task(ds, QC[1], run, sp)
        if t and t[2].lower() == family.lower():
            return sp
    return None


ap = argparse.ArgumentParser()
ap.add_argument('--family', action='append', default=[], help='DATASET=family, e.g. EMBER=lethic')
ap.add_argument('--run', type=int, default=0)
a = ap.parse_args()
FAM = {'EMBER': 'lethic', 'Drebin': 'fakedoc', 'lamda': 'basebridge'}
for x in a.family:
    k, v = x.split('='); FAM[k] = v
out = os.path.join(ROOT, 'figures'); os.makedirs(out, exist_ok=True)
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 9, 'legend.fontsize': 7, 'pdf.fonttype': 42})

# ---------------------------------------------------------------- 1. one setting per dataset
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), constrained_layout=True)
for ax, ds in zip(axes, DS):
    sp = find_split(ds, FAM[ds], a.run)
    if sp is None:
        ax.set_title(f'{DS_NAME[ds]}: family {FAM[ds]} not found'); continue
    n_drift = None
    for lab, fold, col, lw in MODELS[ds]:
        t = load_task(ds, fold, a.run, sp)
        if t is None:
            continue
        s, y, fam = t; n_drift = int(y.sum())
        k, prec, rec, b = curves(s, y)
        ax.plot(k, prec, color=col, lw=lw, ls='-', label=ROLE(lab))
        ax.plot(k, rec, color=col, lw=lw, ls='--')
        ax.plot([k[b]], [prec[b]], marker='|', ms=9, mew=1.5, color=col)
    if n_drift:
        ax.axvline(n_drift, color='0.6', lw=0.8, ls=':')
        ax.set_xlim(0, 2 * n_drift)
        ax.text(n_drift, 0.03, ' effort = 1', color='0.4', fontsize=6)
    ax.set_ylim(0, 1.02); ax.set_xlabel('Inspected samples'); ax.set_title(f'{DS_NAME[ds]} (unseen: {fam})\n' + ', '.join(l for l, *_ in MODELS[ds][2:]), fontsize=8)
    ax.grid(alpha=0.25)
axes[0].set_ylabel('Rate (solid: precision, dashed: recall)')
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 1.0), frameon=False)
fig.savefig(os.path.join(out, 'fig_inspection_examples_proto.pdf'), bbox_inches='tight'); plt.close(fig)

# ---------------------------------------------------------------- 2. mean over all tasks vs. effort
grid = np.linspace(0.02, 2.0, 100); stats = []
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), constrained_layout=True)
for ax, ds in zip(axes, DS):
    for lab, fold, col, lw in MODELS[ds]:
        P, R, E, Pb, Rb = [], [], [], [], []
        for run in range(3):
            for sp in range(8):
                t = load_task(ds, fold, run, sp)
                if t is None:
                    continue
                s, y, _ = t; k, prec, rec, b = curves(s, y); eff = k / max(y.sum(), 1)
                P.append(np.interp(grid, eff, prec)); R.append(np.interp(grid, eff, rec))
                E.append(eff[b]); Pb.append(prec[b]); Rb.append(rec[b])
        if not P:
            continue
        P, R = np.mean(P, 0), np.mean(R, 0)
        ax.plot(grid, P, color=col, lw=lw, ls='-', label=ROLE(lab)); ax.plot(grid, R, color=col, lw=lw, ls='--')
        ax.plot([np.mean(E)], [np.interp(np.mean(E), grid, P)], marker='|', ms=9, mew=1.5, color=col)
        stats.append({'dataset': DS_NAME[ds], 'model': lab, 'tasks': len(E), 'effort_mean': round(float(np.mean(E)), 3),
                      'effort_std': round(float(np.std(E)), 3), 'precision_at_stop': round(float(np.mean(Pb)), 3),
                      'recall_at_stop': round(float(np.mean(Rb)), 3)})
    ax.axvline(1.0, color='0.6', lw=0.8, ls=':')
    ax.set_xlim(0, 2); ax.set_ylim(0, 1.02); ax.set_xlabel('Inspection effort (inspected / drifting)')
    ax.set_title(f'{DS_NAME[ds]} (mean over 24 tasks)\n' + ', '.join(l for l, *_ in MODELS[ds][2:]), fontsize=8); ax.grid(alpha=0.25)
axes[0].set_ylabel('Rate (solid: precision, dashed: recall)')
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 1.0), frameon=False)
fig.savefig(os.path.join(out, 'fig_inspection_mean_proto.pdf'), bbox_inches='tight'); plt.close(fig)

with open(os.path.join(out, 'inspection_stats_proto.csv'), 'w', newline='') as fh:
    if stats:
        w = csv.DictWriter(fh, fieldnames=list(stats[0])); w.writeheader(); w.writerows(stats)
for r in stats:
    print(f"{r['dataset']:7s} {r['model']:20s} tasks {r['tasks']:2d}  effort {r['effort_mean']:.2f}+-{r['effort_std']:.2f}  "
          f"precision {r['precision_at_stop']:.3f}  recall {r['recall_at_stop']:.3f}")
print('Saved figures/fig_inspection_examples_proto.pdf, figures/fig_inspection_mean_proto.pdf, figures/inspection_stats_proto.csv')
