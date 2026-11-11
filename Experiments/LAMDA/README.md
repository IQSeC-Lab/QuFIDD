# LAMDA experiments

All runs use the LAMDA splits in `Data/LAMDA` (PCA-8 input, 8 leave-one-family-out settings,
seeds 42, 123, 456 as `run0`-`run2`). Each model folder holds its training script, SLURM file,
and per-run results (`summary.json`, `ood_scores.csv`, `training_loss.csv`, ...). Model
checkpoints, saved states, and per-run plots are not included.

| Folder | Model in the paper | Original folder (`Experiments2/lamda/pca8/`) |
|---|---|---|
| `models/QuFiDD_pair` | QuFiDD, pairwise loss | `ablation_objective/quantum_hybrid_v7_p702_paironly` |
| `models/QuFiDD_pair+proto` | QuFiDD, pairwise + prototype loss | `quantum_pca8/pennylane/quantum_hybrid_v7_p702` |
| `models/QiCD-{c,p,r}_pair` | QiCD (complex, plain, real), pairwise loss | `ablation_objective/classical_fidtwin_{c,p,r}_702_paironly` |
| `models/QiCD-{c,p,r}_pair+proto` | QiCD, pairwise + prototype loss | `twin_pca8_p702/classical_fidtwin_{c,p,r}_702` |
| `models/CM-f_r+pair`, `models/CM-{e,t,d}_pair` | CADE-minis (full, encoder, tuned, deep), CADE's loss | `cade_pca8_p702/cade_{full,enc,tuned,deep}_8_702` |
| `models/CM-*+proto` | CADE-minis + Euclidean prototype loss (all temperatures) | `cade_pca8_p702_proto/cade_*_8_702_proto` |
| `baselines/PCA_only` | PCA-only baseline (raw and L2) | `pca_only` |
| `analysis/drift_explanation_cade` | CADE's explanation method, all 16 models | `drift_analysis_cade` |
| `analysis/circuit_explanation` | Re-upload decomposition and family mixture | `drift_analysis_quantum` |
| `analysis/fpr_budget` | Ranked F1 under an FPR budget | `fpr_ranked_f1_pca8.py`, `fpr_ranked_f1_summary` |
| `analysis/threshold_sweep` | Threshold sweep | `threshold_sweep_pca8.py`, `threshold_sweep_summary` |
| `analysis/ablation` | Objective ablation helpers | `ablation_objective/*.py, *.slurm, *.sh` |
| `scripts` | Submission and aggregation scripts | `submit_all.sh`, `aggregate_all.slurm`, `run_sweeps.sh` |

The scripts use relative paths of the original layout; run them from a copy of that layout,
or adjust the paths at the top of each script.
