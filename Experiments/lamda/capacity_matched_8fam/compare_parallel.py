#!/usr/bin/env python3
"""
compare_parallel.py -- capacity_matched / capacity_matched_8fam (top level, next to evaluate.py)
======================================================================================================================
Runs the slow part of <pca>/compare_all.py -- the structure-aware statistics (DeLong, per-task bootstrap,
hierarchical bootstrap) -- as ONE TASK PER MODEL PAIR, so they can run in parallel as a SLURM array. It imports
compare_all.py from the PCA folder and uses its own model list, pairs and statistics code, unchanged.

  python3 compare_parallel.py pca8 list                       # how many pairs (with results for both models)
  python3 compare_parallel.py pca8 pair --index 7 --n-boot 500  # statistics for pair 7 only
  python3 compare_parallel.py pca8 merge                      # combine all pairs into the usual files + forest plot

The main tables (overall, paired Wilcoxon tests, per family, calibration, plots) come from
  cd pca8 && python3 compare_all.py --no-stats
Outputs: <pca>/comparison/stats_parts/pair_XXX.{txt,csv files}; after merge:
  <pca>/comparison/structure_statistics.txt, delong_per_task.csv, bootstrap_per_task.csv, hierarchical_bootstrap.csv,
  plots/hierarchical_bootstrap_forest.png
"""
import os, sys, csv, glob, argparse, importlib.util
import numpy as np


def load_compare(pca_dir):
    path = os.path.join(os.path.abspath(pca_dir), 'compare_all.py')
    spec = importlib.util.spec_from_file_location('compare_all', path); m = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, [path]; spec.loader.exec_module(m); sys.argv = argv
    return m


def build(m):
    data = {}
    for name, rel in m.MODELS:
        rows = m.collect(os.path.join(m.FE, rel))
        if rows:
            data[name] = {(x['run'], x['split']): x for x in rows}
    fam_of = {k: row['family'] for n in data for k, row in data[n].items()}
    pairs = []
    for A, B in m.KEY_PAIRS:
        if A in data and B in data and set(data[A]) & set(data[B]) and (A, B) not in pairs:
            pairs.append((A, B))
    return data, fam_of, pairs


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('pca_dir'); ap.add_argument('cmd', choices=['list', 'pair', 'merge'])
    ap.add_argument('--index', type=int); ap.add_argument('--n-boot', type=int, default=500)
    a = ap.parse_args()
    m = load_compare(a.pca_dir); data, fam_of, pairs = build(m)
    parts = os.path.join(m.OUTD, 'stats_parts'); os.makedirs(parts, exist_ok=True)
    if a.cmd == 'list':
        for i, (A, B) in enumerate(pairs):
            print(f'{i:3d}  {A}  vs  {B}')
        print(f'N_PAIRS={len(pairs)}')
        return
    if a.cmd == 'pair':
        A, B = pairs[a.index]; d = os.path.join(parts, f'pair_{a.index:03d}'); os.makedirs(d, exist_ok=True)
        lines = []
        def log(s=''):
            print(s, flush=True); lines.append(s)
        m.run_statistics(data, list(data), fam_of, a.n_boot, [(A, B)], log, d, seed=a.index)
        open(os.path.join(d, 'log.txt'), 'w').write('\n'.join(lines) + '\n')
        return
    # merge
    done = sorted(glob.glob(os.path.join(parts, 'pair_*', 'log.txt')))
    missing = [i for i in range(len(pairs)) if not os.path.exists(os.path.join(parts, f'pair_{i:03d}', 'log.txt'))]
    lines = ['=' * 118, f'STRUCTURE-AWARE STATISTICS  ({len(done)} of {len(pairs)} pairs)',
             '  DeLong: per task, same test samples, p < 0.05. Per-family string: one character per seed (A = A sig. better,'
             ' B = B sig. better, - = n.s.).', '  Boot: per task, paired bootstrap over test samples, 95% CI of the difference.',
             '  Hierarchical bootstrap: families -> seeds -> test samples resampled; the CI accounts for all three levels.', '=' * 118]
    for f in done:
        lines += open(f).read().rstrip('\n').split('\n')
    if missing:
        lines += ['', f'MISSING pairs (not finished or failed): {missing}']
    open(os.path.join(m.OUTD, 'structure_statistics.txt'), 'w').write('\n'.join(lines) + '\n')
    rows_h = []
    for fn in ('delong_per_task.csv', 'bootstrap_per_task.csv', 'hierarchical_bootstrap.csv'):
        hdr, rows = None, []
        for pdir in sorted(glob.glob(os.path.join(parts, 'pair_*'))):
            p = os.path.join(pdir, fn)
            if os.path.exists(p):
                r = list(csv.reader(open(p))); hdr = r[0]; rows += r[1:]
        if hdr:
            with open(os.path.join(m.OUTD, fn), 'w', newline='') as fh:
                w = csv.writer(fh); w.writerow(hdr); w.writerows(rows)
        if fn == 'hierarchical_bootstrap.csv':
            rows_h = [[r[0], r[1], r[2], float(r[3]), float(r[4]), float(r[5])] for r in rows]
    if rows_h:                                            # forest plot, as compare_all.py draws it
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        mets = list(dict.fromkeys(r[2] for r in rows_h)); labels = list(dict.fromkeys(f'{r[0]} vs {r[1]}' for r in rows_h))
        fig, ax = plt.subplots(1, len(mets), figsize=(4.2 * len(mets), 0.32 * len(labels) + 2.2), sharey=True, squeeze=False)
        for a_, nm in zip(ax[0], mets):
            rs = [r for r in rows_h if r[2] == nm]; yy = [labels.index(f'{r[0]} vs {r[1]}') for r in rs]
            a_.errorbar([r[3] for r in rs], yy, xerr=[[max(0.0, r[3] - r[4]) for r in rs], [max(0.0, r[5] - r[3]) for r in rs]], fmt='o', capsize=3)
            a_.axvline(0, color='k', lw=1); a_.set_title(nm, fontsize=10); a_.grid(alpha=0.3)
            a_.set_yticks(range(len(labels))); a_.set_yticklabels(labels, fontsize=7); a_.invert_yaxis()
        fig.suptitle('Hierarchical bootstrap: mean difference (A minus B) with 95% CI', fontsize=10); fig.tight_layout()
        os.makedirs(os.path.join(m.OUTD, 'plots'), exist_ok=True)
        fig.savefig(os.path.join(m.OUTD, 'plots', 'hierarchical_bootstrap_forest.png'), dpi=140); plt.close(fig)
    print(f'merged {len(done)} of {len(pairs)} pairs -> {m.OUTD}/structure_statistics.txt'
          + (f'   MISSING: {missing}' if missing else ''))


if __name__ == '__main__':
    main()
