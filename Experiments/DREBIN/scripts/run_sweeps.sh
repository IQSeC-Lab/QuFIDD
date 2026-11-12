#!/bin/bash
# Submit both sweeps for this dataset (run after all models have finished). Results: threshold_sweep_summary/ and fpr_ranked_f1_summary/
cd "$(dirname "$0")"; mkdir -p logs
j=$(sbatch --parsable run_threshold_sweep_pca8.slurm)
sbatch --dependency=afterok:$j run_fpr_ranked_f1_pca8.slurm   # needs threshold_sweep_pca8.py's functions; runs after the sweep
