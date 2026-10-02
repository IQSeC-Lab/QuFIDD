#!/bin/bash
#SBATCH --job-name=G16CADE
#SBATCH --partition=main
#SBATCH --array=0-11
#SBATCH --output=logs/g16_%A_%a.out
#SBATCH --error=logs/g16_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=288:00:00

# CADE exactly as released:        sbatch run_Group16_cade_original_4class.slurm
# Matched protocol (G14/17/18):    sbatch --export=ALL,EXTRA="--protocol matched",OUT=results_matched run_Group16_cade_original_4class.slurm
EXTRA=${EXTRA:-}
OUT=${OUT:-results}

echo "================================================"
echo "Group16: CADE original (no PCA) -- 4-class LAMDA"
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

WORK_DIR=/home/snowm001/projects/OOD/Experiments/lamda/Group16
cd $WORK_DIR
mkdir -p logs $OUT

python3 Group16_cade_original_4class.py $SLURM_ARRAY_TASK_ID $EXTRA \
    --data-dir /home/snowm001/projects/OOD/Data/cade_lamda_4class_vt \
    --prefix drebin_new \
    --results-dir $WORK_DIR/$OUT

echo "================================================"
echo "End time: $(date)"
echo "================================================"
