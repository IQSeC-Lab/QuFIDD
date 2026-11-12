#!/bin/bash
#SBATCH --job-name=plot_drift
#SBATCH --partition=main
#SBATCH --time=288:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/plot_drift_%j.out
# Re-upload decomposition figure for the Drift Analysis section.
# Run from Experiments2 (where DX_reupload.csv lives):  mkdir -p logs figures/drift && sbatch run_plot_drift.slurm
# Output: figures/drift/reupload_decomposition.{pdf,png} -> copy the PDF to imgs/Evaluation_drift/ in the paper.

# Activate the same environment as run_plot_score_dist.slurm (copy its conda/module lines here if they differ).

source /home/snowm001/projects/OOD/miniconda3/bin/activate OOD_val
cd "$SLURM_SUBMIT_DIR"
python3 plot_drift_analysis.py --root . --out figures/drift
