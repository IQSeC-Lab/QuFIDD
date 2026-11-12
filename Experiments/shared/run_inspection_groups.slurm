#!/bin/bash
#SBATCH --job-name=insp_grp
#SBATCH --partition=main
#SBATCH --output=logs/insp_grp_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=288:00:00
# Submit from Experiments2:  mkdir -p logs && sbatch run_inspection_groups.slurm
#   -> INSPECTION_GROUPS.csv, fig_inspection_tikz.tex
source /home/snowm001/projects/OOD/miniconda3/bin/activate OOD_val
cd "$SLURM_SUBMIT_DIR"
python3 inspection_groups.py
