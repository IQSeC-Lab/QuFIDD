#!/bin/bash
#SBATCH --job-name=plot_sdist
#SBATCH --partition=main
#SBATCH --output=plot_score_dist_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=288:00:00
# Submit from Experiments2:  sbatch run_plot_score_dist.slurm
#   -> figures/fig_score_dist.pdf, figures/fig_score_dist_3rows.pdf, figures/score_dist_stats.csv
source /home/snowm001/projects/OOD/miniconda3/bin/activate OOD_val
cd "$SLURM_SUBMIT_DIR"
python3 plot_score_dist.py
