#!/bin/bash
#SBATCH --job-name=plot_tsne
#SBATCH --partition=main
#SBATCH --output=plot_tsne_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=288:00:00
# Submit from Experiments2:  sbatch run_plot_tsne.slurm
#   -> figures/fig_tsne_ember_zbot.pdf, figures/tsne_ember_zbot_coords.csv
source /home/snowm001/projects/OOD/miniconda3/bin/activate OOD_val
cd "$SLURM_SUBMIT_DIR"
python3 plot_tsne.py --ds EMBER --split 1 --run 2
