#!/bin/bash
#SBATCH --job-name=perfam_grp
#SBATCH --partition=main
#SBATCH --output=logs/perfam_grp_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=288:00:00
# Submit from Experiments2:  mkdir -p logs && sbatch run_perfamily_groups.slurm
#   -> PERFAMILY_GROUPS.csv
source /home/snowm001/projects/OOD/miniconda3/bin/activate OOD_val
cd "$SLURM_SUBMIT_DIR"
python3 perfamily_groups.py
