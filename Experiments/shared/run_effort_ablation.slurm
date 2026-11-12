#!/bin/bash
#SBATCH --job-name=effort_abl
#SBATCH --partition=main
#SBATCH --output=logs/effort_abl_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=288:00:00
# Submit from Experiments2:  mkdir -p logs && sbatch run_effort_ablation.slurm
#   -> logs/effort_abl_<job>.out and EFFORT_ABLATION.csv
source /home/snowm001/projects/OOD/miniconda3/bin/activate OOD_val
cd "$SLURM_SUBMIT_DIR"
python3 effort_ablation.py
