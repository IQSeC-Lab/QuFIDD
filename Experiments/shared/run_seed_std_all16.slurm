#!/bin/bash
#SBATCH --job-name=seed_std
#SBATCH --partition=main
#SBATCH --output=logs/seed_std_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=288:00:00
# Submit from Experiments2:  mkdir -p logs && sbatch run_seed_std_all16.slurm
#   -> SEED_STD_ALL16.csv
source /home/snowm001/projects/OOD/miniconda3/bin/activate OOD_val
cd "$SLURM_SUBMIT_DIR"
python3 seed_std_all16.py
