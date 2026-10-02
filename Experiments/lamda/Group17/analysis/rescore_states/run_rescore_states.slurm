#!/bin/bash
#SBATCH --job-name=G17Rescore
#SBATCH --partition=main
#SBATCH --array=0-11
#SBATCH --output=logs/g17rescore_%A_%a.out
#SBATCH --error=logs/g17rescore_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=288:00:00

# Trained weights:    sbatch run_rescore_states.slurm
# Untrained weights:  sbatch --export=ALL,EXTRA=--untrained run_rescore_states.slurm
EXTRA=${EXTRA:-}

echo "================================================"
echo "Group17 rescore_states ${EXTRA:-(trained weights)}"
echo "Job ID:     $SLURM_JOB_ID"
echo "Array Task: $SLURM_ARRAY_TASK_ID"
echo "Node:       $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "================================================"

source /home/snowm001/projects/OOD/miniconda3/bin/activate OOD_env

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

WORK_DIR=/home/snowm001/projects/OOD/Experiments/lamda/Group17/analysis/rescore_states
cd $WORK_DIR
mkdir -p logs

python3 rescore_states.py $SLURM_ARRAY_TASK_ID $EXTRA \
    --data-dir /home/snowm001/projects/OOD/Data/cade_lamda_4class_vt \
    --prefix drebin_new

echo "================================================"
echo "End time: $(date)"
echo "================================================"
