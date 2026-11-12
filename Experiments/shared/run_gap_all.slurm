#!/bin/bash
#SBATCH --job-name=gap_all
#SBATCH --partition=main
#SBATCH --output=logs/gap_all_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=288:00:00
# Submit from Experiments2:  mkdir -p logs && sbatch run_gap_all.slurm
#   -> GAP_ALL.csv
source /home/snowm001/projects/OOD/miniconda3/bin/activate OOD_val
cd "$SLURM_SUBMIT_DIR"
python3 gap_all.py
