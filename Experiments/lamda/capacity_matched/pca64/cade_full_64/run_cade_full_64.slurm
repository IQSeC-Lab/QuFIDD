#!/bin/bash
#SBATCH --job-name=pca64_cade_full_64
#SBATCH --partition=main
#SBATCH --array=0-11
#SBATCH --output=logs/test_%A_%a.out
#SBATCH --error=logs/test_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=288:00:00
# 12 test tasks = 4 held-out families x 3 seeds (42, 123, 456)
source /home/snowm001/projects/OOD/miniconda3/bin/activate OOD_env
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
WORK_DIR=/home/snowm001/projects/OOD/Experiments/lamda/capacity_matched/pca64/cade_full_64
cd $WORK_DIR
mkdir -p logs results validation
python3 run_cade_full_64.py $SLURM_ARRAY_TASK_ID \
    --data-dir /home/snowm001/projects/OOD/Data/cade_lamda_4class_vt --prefix drebin_new --results-dir $WORK_DIR/results
