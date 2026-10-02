#!/bin/bash
#SBATCH --job-name=P8_quantum_v2_8
#SBATCH --partition=main
#SBATCH --array=0-3
#SBATCH --output=logs/quantum_v2_8_%A_%a.out
#SBATCH --error=logs/quantum_v2_8_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=288:00:00
# Pilot = seed 42, 4 families (tasks 0-3). Seeds 123/456: sbatch --array=4-11 run_quantum_v2_8.slurm
source /home/snowm001/projects/OOD/miniconda3/bin/activate OOD_env
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
WORK_DIR=/home/snowm001/projects/OOD/Experiments/lamda/pca8_experiment/quantum_v2_8
cd $WORK_DIR
mkdir -p logs results
python3 run_quantum_v2_8.py $SLURM_ARRAY_TASK_ID \
    --data-dir /home/snowm001/projects/OOD/Data/cade_lamda_4class_vt --prefix drebin_new --results-dir $WORK_DIR/results
