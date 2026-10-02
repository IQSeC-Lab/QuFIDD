#!/bin/bash
#SBATCH --job-name=G20CADEm
#SBATCH --partition=main
#SBATCH --array=0-11
#SBATCH --output=logs/g20_%A_%a.out
#SBATCH --error=logs/g20_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=288:00:00

# Default (CADE checkpointing):  sbatch run_Group20_cade_matched_pca5_4class.slurm
# Matched stopping rule (as Group18): sbatch --export=ALL,EXTRA="--protocol matched",OUT=results_matched run_Group20_cade_matched_pca5_4class.slurm
EXTRA=${EXTRA:-}
OUT=${OUT:-results}

echo "================================================"
echo "Group20: CADE-matched on PCA-5 (5-11-3, 206 params, lr 0.001) -- 4-class LAMDA"
echo "Extra args: ${EXTRA:-none}   Results dir: $OUT"
echo "Job ID:     $SLURM_JOB_ID"
echo "Array Task: $SLURM_ARRAY_TASK_ID"
echo "Node:       $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "================================================"

source /home/snowm001/projects/OOD/miniconda3/bin/activate OOD_env

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

WORK_DIR=/home/snowm001/projects/OOD/Experiments/lamda/Group20
cd $WORK_DIR
mkdir -p logs $OUT

python3 Group20_cade_matched_pca5_4class.py $SLURM_ARRAY_TASK_ID $EXTRA \
    --data-dir /home/snowm001/projects/OOD/Data/cade_lamda_4class_vt \
    --prefix drebin_new \
    --results-dir $WORK_DIR/$OUT

echo "================================================"
echo "End time: $(date)"
echo "================================================"
