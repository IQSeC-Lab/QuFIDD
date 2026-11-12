#!/bin/bash
#SBATCH --job-name=reup_onerow
#SBATCH --partition=main
#SBATCH --time=288:00:00
#SBATCH --output=reup_onerow_%j.out
source /home/snowm001/projects/OOD/miniconda3/bin/activate OOD_val
cd ~/projects/OOD/Experiments2
python draw_reupload_onerow.py
