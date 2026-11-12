#!/bin/bash
#SBATCH --job-name=dr_fprF1
#SBATCH --partition=main
#SBATCH --output=logs/fpr_ranked_f1_%j.out
#SBATCH --error=logs/fpr_ranked_f1_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=288:00:00
# Submit from Drebin/pca8/:  mkdir -p logs && sbatch run_fpr_ranked_f1_pca8.slurm
# extra arguments are passed on, e.g.  sbatch run_fpr_ranked_f1_pca8.slurm --include-ablation
source /home/snowm001/projects/OOD/miniconda3/bin/activate OOD_env
cd "$SLURM_SUBMIT_DIR"
python3 fpr_ranked_f1_pca8.py "$@"
