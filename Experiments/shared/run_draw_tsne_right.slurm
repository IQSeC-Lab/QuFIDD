#!/bin/bash
#SBATCH --job-name=tsne_right
#SBATCH --partition=main
#SBATCH --time=288:00:00
#SBATCH --output=tsne_right_%j.out
source /home/snowm001/projects/OOD/miniconda3/bin/activate OOD_val
cd ~/projects/OOD/Experiments2
python draw_tsne_failure_right.py figures/tsne_failure_coords.csv figures/fig_tsne_failure_right
