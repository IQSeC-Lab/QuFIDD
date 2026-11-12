#!/bin/bash
#SBATCH --job-name=fpr_all16
#SBATCH --partition=main
#SBATCH --output=logs/fpr_all16_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=288:00:00
# Submit from Experiments2:  mkdir -p logs && sbatch run_fpr_budget_all16.slurm   -> FPR_BUDGET_ALL16.csv
source /home/snowm001/projects/OOD/miniconda3/bin/activate OOD_val
cd "$SLURM_SUBMIT_DIR"
python3 fpr_budget_all16.py
