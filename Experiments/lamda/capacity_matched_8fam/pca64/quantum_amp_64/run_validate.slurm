#!/bin/bash
#SBATCH --job-name=pca64_quantum_amp_64_val
#SBATCH --partition=main
#SBATCH --array=0-23
#SBATCH --output=logs/validate_%A_%a.out
#SBATCH --error=logs/validate_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=288:00:00
# VALIDATION jobs (unused in this study)
source /home/snowm001/projects/OOD/miniconda3/bin/activate OOD_env
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
WORK_DIR=/home/snowm001/projects/OOD/Experiments/lamda/capacity_matched_8fam/pca64/quantum_amp_64
cd $WORK_DIR
mkdir -p logs results validation
python3 run_quantum_amp_64.py validate --job $SLURM_ARRAY_TASK_ID \
    --data-dir /home/snowm001/projects/OOD/Data/cade_lamda --prefix drebin_new
