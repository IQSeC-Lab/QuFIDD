#!/bin/bash
#SBATCH --job-name=tsne_all
#SBATCH --partition=main
#SBATCH --output=tsne_all_%A_%a.out
#SBATCH --array=0-7
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=288:00:00
# Submit from Experiments2:  sbatch run_plot_tsne_all.slurm
#   one task per EMBER split (same seed run 2 = 456 and same t-SNE settings for all)
#   -> figures/fig_tsne_ember_<family>.pdf, figures/tsne_ember_<family>_coords.csv
source /home/snowm001/projects/OOD/miniconda3/bin/activate OOD_val
cd "$SLURM_SUBMIT_DIR"
python3 plot_tsne.py --ds EMBER --split $SLURM_ARRAY_TASK_ID --run 2
