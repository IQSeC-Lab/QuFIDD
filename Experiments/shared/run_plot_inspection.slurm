#!/bin/bash
#SBATCH --job-name=plot_insp
#SBATCH --partition=main
#SBATCH --output=plot_inspection_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=288:00:00
# Submit from Experiments2:  sbatch run_plot_inspection.slurm   -> figures/fig_inspection_examples.pdf, fig_inspection_mean.pdf
source /home/snowm001/projects/OOD/miniconda3/bin/activate OOD_val
cd "$SLURM_SUBMIT_DIR"
python3 plot_inspection.py --family EMBER=lethic --family Drebin=fakedoc --family lamda=basebridge --run 0
