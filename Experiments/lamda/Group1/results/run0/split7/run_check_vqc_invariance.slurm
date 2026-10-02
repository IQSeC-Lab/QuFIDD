#!/bin/bash
#SBATCH --job-name=VQCInv
#SBATCH --partition=main
#SBATCH --output=check_vqc_invariance_%j.out
#SBATCH --error=check_vqc_invariance_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00

echo "================================================"
echo "VQC unitary-invariance check (M1c, trained vs fresh q_weights)"
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "================================================"

bash
source /home/snowm001/projects/OOD/miniconda3/bin/activate OOD_env

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

python3 check_vqc_invariance_direct.py \
    "." \
    "/home/snowm001/projects/OOD/Experiments/lamda/Group1/Group1_cade_comp_fidelity_5q.py" \
    "/home/snowm001/projects/OOD/Data/cade_lamda" \
    "drebin_new"

echo "================================================"
echo "End time: $(date)"
echo "================================================"
