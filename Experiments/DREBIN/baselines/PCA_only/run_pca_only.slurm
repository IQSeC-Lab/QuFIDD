#!/bin/bash
#SBATCH --job-name=dr_p8_pca_only
#SBATCH --partition=main

#SBATCH --output=logs/pca_only_%j.out
#SBATCH --error=logs/pca_only_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=288:00:00
source /home/snowm001/projects/OOD/miniconda3/bin/activate OOD_env
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
WORK_DIR=$SLURM_SUBMIT_DIR
cd $WORK_DIR
mkdir -p logs results
python3 run_pca_only.py all --data-dir /home/snowm001/projects/OOD/Data2/Drebin/cade_drebin --prefix drebin_new
