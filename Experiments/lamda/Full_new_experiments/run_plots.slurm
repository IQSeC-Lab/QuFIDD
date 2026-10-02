#!/bin/bash
#SBATCH --job-name=FN_plots
#SBATCH --partition=main
#SBATCH --output=plots_%j.out
#SBATCH --error=plots_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=288:00:00

source /home/snowm001/projects/OOD/miniconda3/bin/activate OOD_env
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
cd /home/snowm001/projects/OOD/Experiments/lamda/Full_new_experiments

python3 plot_latent_spaces.py all --max-train 1200
cd quantum && python3 plot_feature_maps.py all
echo "done $(date)"
