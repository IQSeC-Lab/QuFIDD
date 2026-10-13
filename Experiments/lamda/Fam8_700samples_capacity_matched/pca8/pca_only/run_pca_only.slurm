#!/bin/bash
#SBATCH --job-name=pca8_only_700
#SBATCH --partition=main
#SBATCH --output=logs/pca_only_%j.out
#SBATCH --error=logs/pca_only_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=288:00:00
# Zero-parameter PCA-8 baselines, all 24 tasks + aggregate (no training)
source /home/snowm001/projects/OOD/miniconda3/bin/activate OOD_env
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
WORK_DIR=/home/snowm001/projects/OOD/Experiments/lamda/Fam8_700samples_capacity_matched/pca8/pca_only
cd $WORK_DIR
mkdir -p logs
python3 run_pca_only.py all --data-dir /home/snowm001/projects/OOD/Data/cade_lamda_700_vt --prefix drebin_new
