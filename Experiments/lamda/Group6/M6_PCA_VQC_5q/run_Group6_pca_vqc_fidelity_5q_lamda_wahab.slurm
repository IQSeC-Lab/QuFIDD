#!/bin/bash
#SBATCH --job-name=G6LAMDA
#SBATCH --partition=main
#SBATCH --array=0-23
#SBATCH --output=/home/snowm001/projects/OOD/Experiments/lamda/Group6/M6_PCA_VQC_5q/logs/g6_%A_%a.out
#SBATCH --error=/home/snowm001/projects/OOD/Experiments/lamda/Group6/M6_PCA_VQC_5q/logs/g6_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=288:00:00

echo "================================================"
echo "Group6: PCA + VQC + Fidelity Loss 5Q — LAMDA"
echo "Scoring: M1a=Euclidean M1b=Cosine M1c=Fidelity"
echo "Families: BaseBridge DroidKungFu Kmin Opfake"
echo "          Plankton Airpush Smsreg Fakeapp"
echo "Job ID:     $SLURM_JOB_ID"
echo "Array Task: $SLURM_ARRAY_TASK_ID"
echo "Node:       $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "================================================"

source /home/snowm001/projects/OOD/miniconda3/bin/activate OOD_env

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

WORK_DIR=/home/snowm001/projects/OOD/Experiments/lamda/Group6/M6_PCA_VQC_5q
cd $WORK_DIR
mkdir -p logs results

python3 Group6_pca_vqc_fidelity_5q.py $SLURM_ARRAY_TASK_ID \
    --data-dir /home/snowm001/projects/OOD/Data/cade_lamda \
    --prefix drebin_new \
    --results-dir $WORK_DIR/results

echo "================================================"
echo "End time: $(date)"
echo "================================================"
