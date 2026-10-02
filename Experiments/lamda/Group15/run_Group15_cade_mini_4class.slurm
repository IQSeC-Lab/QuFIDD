#!/bin/bash
#SBATCH --job-name=G15Mini
#SBATCH --partition=main
#SBATCH --array=0-11
#SBATCH --output=logs/g15_%A_%a.out
#SBATCH --error=logs/g15_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00

echo "================================================"
echo "Group15: CADE-mini (215 params, 4-class, frozen PCA) — LAMDA"
echo "Families: droidkungfu plankton smsreg airpush"
echo "Job ID:     $SLURM_JOB_ID"
echo "Array Task: $SLURM_ARRAY_TASK_ID"
echo "Node:       $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "================================================"

bash
source /home/snowm001/projects/OOD/miniconda3/bin/activate OOD_env

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

WORK_DIR=/home/snowm001/projects/OOD/Experiments/lamda/Group15
cd $WORK_DIR
mkdir -p logs results

python3 Group15_cade_mini_4class.py $SLURM_ARRAY_TASK_ID \
    --data-dir /home/snowm001/projects/OOD/Data/cade_lamda_4class_vt \
    --prefix drebin_new \
    --results-dir $WORK_DIR/results

echo "================================================"
echo "End time: $(date)"
echo "================================================"
