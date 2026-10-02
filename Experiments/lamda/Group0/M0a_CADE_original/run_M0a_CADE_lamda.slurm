#!/bin/bash
#SBATCH --job-name=M0aLAMDA
#SBATCH --output=logs/m0a_%A_%a.out
#SBATCH --error=logs/m0a_%A_%a.err
#SBATCH --partition=main
#SBATCH --time=288:00:00
#SBATCH --array=0-23
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

bash
source ~/projects/OOD/miniconda3/bin/activate OOD_env

echo "================================================"
echo "M0a CADE Original + M0b Threshold-Free — LAMDA (monthwise)"
echo "Job ID:     $SLURM_ARRAY_JOB_ID"
echo "Array Task: $SLURM_ARRAY_TASK_ID"
echo "Node:       $(hostname)"
echo "Start time: $(date)"
echo "================================================"

mkdir -p logs results

python3 M0a_CADE_original.py $SLURM_ARRAY_TASK_ID \
    --data-dir ~/projects/OOD/Data/cade_lamda \
    --prefix drebin_new \
    --results-dir ./results

echo "================================================"
echo "End time: $(date)"
echo "================================================"
