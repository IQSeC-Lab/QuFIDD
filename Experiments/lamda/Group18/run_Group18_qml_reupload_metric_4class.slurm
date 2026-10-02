#!/bin/bash
#SBATCH --job-name=G18QML
#SBATCH --partition=main
#SBATCH --array=0-11
#SBATCH --output=logs/g18_%A_%a.out
#SBATCH --error=logs/g18_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=288:00:00

# Pilot (run 0 = seed 42, all 4 splits):  sbatch --array=0-3 run_Group18_qml_reupload_metric_4class.slurm
# Full (12 tasks):                        sbatch run_Group18_qml_reupload_metric_4class.slurm
# Untrained ablation (no training):       sbatch --export=ALL,EXTRA="--max-epochs 0",OUT=ablation/untrained/results run_Group18_qml_reupload_metric_4class.slurm
EXTRA=${EXTRA:-}
OUT=${OUT:-results}

echo "================================================"
echo "Group18: re-uploading quantum metric learning (5q, 20 layers, 4 uploads) -- 4-class LAMDA"
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

WORK_DIR=/home/snowm001/projects/OOD/Experiments/lamda/Group18
cd $WORK_DIR
mkdir -p logs $OUT

python3 Group18_qml_reupload_metric_4class.py $SLURM_ARRAY_TASK_ID $EXTRA \
    --data-dir /home/snowm001/projects/OOD/Data/cade_lamda_4class_vt \
    --prefix drebin_new \
    --results-dir $WORK_DIR/$OUT

echo "================================================"
echo "End time: $(date)"
echo "================================================"
