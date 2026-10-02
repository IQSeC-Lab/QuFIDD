# ---- 1. run_quantum_amp_8.py
sed -i \
 -e 's/^N_SPLITS       = 4$/N_SPLITS       = 8/' \
 -e "s#'~/projects/OOD/Data/cade_lamda_4class_vt'#'~/projects/OOD/Data/cade_lamda'#" \
 -e 's/^FLAG_ALL_F1    = 0.769 .*/FLAG_ALL_F1    = None           # flag-all F1 floor: computed per task from its test set (2p\/(1+p), p = OOD share)/' \
 -e "s/^\(                \)row = {'run': r, 'split': s, 'family': d\['ood_family'\], 'n_params': d.get('n_trainable_params', 0),/\1p_ood = d.get('n_ood', 0) \/ max(d.get('n_test', 1), 1)\n&/" \
 -e "s/^\(                       'best_epoch': d.get('best_epoch', 0)\)}$/\1, 'floor': 2 * p_ood \/ (1 + p_ood)}/" \
 -e "s/^    log(f'  Threshold-free F1 flag-all floor = {FLAG_ALL_F1}. std = population std.')$/    floor = float(np.mean([x['floor'] for x in rows]))\n    log(f'  Threshold-free F1 flag-all floor = {floor:.3f} (mean over tasks). std = population std.')/" \
 -e "s/^    a.axhline(FLAG_ALL_F1, color='k', ls=':', lw=1, label='flag-all F1 (T=0 floor)')$/    a.axhline(floor, color='k', ls=':', lw=1, label='flag-all F1 (T=0 floor)')/" \
 -e "s/fieldnames=\['run', 'split', 'family', 'n_params', 'best_epoch'\] + keys/fieldnames=['run', 'split', 'family', 'n_params', 'best_epoch', 'floor'] + keys/" \
 -e 's#capacity_matched/pca8 / quantum_amp_8#capacity_matched_8fam/pca8 / quantum_amp_8#' \
 pca8/quantum_amp_8/run_quantum_amp_8.py
# ---- 2. run_quantum_amp_8.slurm
sed -i \
 -e 's/^#SBATCH --array=0-11$/#SBATCH --array=0-23/' \
 -e 's#lamda/capacity_matched/pca8/quantum_amp_8#lamda/capacity_matched_8fam/pca8/quantum_amp_8#' \
 -e 's#OOD/Data/cade_lamda_4class_vt#OOD/Data/cade_lamda#' \
 -e 's/^# 12 test tasks = 4 held-out families x 3 seeds (42, 123, 456)$/# 24 test tasks = 8 held-out families x 3 seeds (42, 123, 456)/' \
 pca8/quantum_amp_8/run_quantum_amp_8.slurm
# ---- 3. compare_all.py
sed -i \
 -e 's/^N_SPLITS       = 4$/N_SPLITS       = 8/' \
 -e "s#'~/projects/OOD/Data/cade_lamda_4class_vt'#'~/projects/OOD/Data/cade_lamda'#" \
 -e 's/^FLAG_ALL_F1    = 0.769 .*/def _flag_all_floor():\n    """Mean flag-all F1 over the tasks found (2p \/ (1 + p), p = OOD share of each test set)."""\n    import glob as _g\n    v = []\n    for f in _g.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "*", "results", "run*", "split*", "summary.json")):\n        d = json.load(open(f)); p = d.get("n_ood", 0) \/ max(d.get("n_test", 1), 1); v.append(2 * p \/ (1 + p))\n    return float(np.mean(v)) if v else float("nan")\nFLAG_ALL_F1    = _flag_all_floor()      # 8-family test sets differ in size: computed from the saved summaries/' \
 -e 's/^# The 12 tasks are 4 held-out families x 3 seeds on the SAME data, so they are not independent.$/# The 24 tasks are 8 held-out families x 3 seeds on the SAME data, so they are not independent./' \
 -e "s/n = 12 -> smallest possible two-sided Wilcoxon p ~ 0.0005; n = 4 (one seed) -> 0.125./n = 24 -> smallest possible two-sided Wilcoxon p ~ 1e-7; n = 8 (one seed) -> 0.008./" \
 -e 's/CADE-mini full 8-8-3 + decoder/CADE-mini full 8-6-7 + decoder/g' \
 -e 's/CADE-mini encoder-only 8-16-3/CADE-mini encoder-only 8-12-7/g' \
 -e 's/CADE-mini tuned 8-16-3 margin 20/CADE-mini tuned 8-12-7 margin 20/g' \
 -e 's#capacity_matched/pca8#capacity_matched_8fam/pca8#g' \
 pca8/compare_all.py
