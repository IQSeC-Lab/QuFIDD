#!/bin/bash
#SBATCH --job-name=pca8_v4val
#SBATCH --partition=main
#SBATCH --array=0-143
#SBATCH --output=logs/val_%A_%a.out
#SBATCH --error=logs/val_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=288:00:00
# 144 VALIDATION jobs = 6 loss-weight settings x 24 folds (training data only)
source /home/snowm001/projects/OOD/miniconda3/bin/activate OOD_env
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
WORK_DIR=/home/snowm001/projects/OOD/Experiments/lamda/capacity_matched_8fam/pca8/quantum_hybrid_v4
cd $WORK_DIR
mkdir -p logs results validation
python3 run_quantum_hybrid_v4.py validate --job $SLURM_ARRAY_TASK_ID \
    --data-dir /home/snowm001/projects/OOD/Data/cade_lamda --prefix drebin_new
