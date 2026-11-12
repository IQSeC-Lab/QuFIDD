#!/bin/bash
#SBATCH --job-name=reup_tikz
#SBATCH --partition=main
#SBATCH --time=288:00:00
#SBATCH --output=reup_tikz_%j.out
source /home/snowm001/projects/OOD/miniconda3/bin/activate OOD_val
cd ~/projects/OOD/Experiments2
python gen_reupload_tikz.py
