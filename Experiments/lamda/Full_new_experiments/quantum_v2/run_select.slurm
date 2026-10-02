#!/bin/bash
#SBATCH --job-name=FN_qv2_sel
#SBATCH --partition=main
#SBATCH --array=0-71
#SBATCH --output=logs/select_%A_%a.out
#SBATCH --error=logs/select_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=288:00:00

# 72 VALIDATION jobs = 6 variants x 4 splits x 3 pseudo-OOD folds (training data only; test never used)
echo "quantum_v2 selection job $SLURM_ARRAY_TASK_ID  node $SLURMD_NODENAME  start $(date)"
source /home/snowm001/projects/OOD/miniconda3/bin/activate OOD_env
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd /home/snowm001/projects/OOD/Experiments/lamda/Full_new_experiments/quantum_v2
mkdir -p logs selection
python3 run_quantum_v2.py select --job $SLURM_ARRAY_TASK_ID \
    --data-dir /home/snowm001/projects/OOD/Data/cade_lamda_4class_vt --prefix drebin_new
echo "end $(date)"
