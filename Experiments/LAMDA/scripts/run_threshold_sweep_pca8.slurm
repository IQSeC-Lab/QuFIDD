#!/bin/bash
#SBATCH --job-name=lm_sweep
#SBATCH --partition=main
#SBATCH --output=logs/threshold_sweep_%j.out
#SBATCH --error=logs/threshold_sweep_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=288:00:00
# Submit from lamda/pca8/:  mkdir -p logs && sbatch run_threshold_sweep_pca8.slurm
# extra arguments are passed on, e.g.  sbatch run_threshold_sweep_pca8.slurm --include-ablation
source /home/snowm001/projects/OOD/miniconda3/bin/activate OOD_env
cd "$SLURM_SUBMIT_DIR"
python3 threshold_sweep_pca8.py "$@"
