#!/bin/bash
#SBATCH --job-name=P64_cade_mini_64
#SBATCH --partition=main
#SBATCH --array=0-3
#SBATCH --output=logs/test_%A_%a.out
#SBATCH --error=logs/test_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=288:00:00
source /home/snowm001/projects/OOD/miniconda3/bin/activate OOD_env
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
WORK_DIR=/home/snowm001/projects/OOD/Experiments/lamda/pca64_experiment/cade_mini_64
cd $WORK_DIR
mkdir -p logs results validation
# Test pilot = seed 42, 4 families (tasks 0-3). Seeds 123/456: sbatch --array=4-11 run_cade_mini_64.slurm
python3 run_cade_mini_64.py $SLURM_ARRAY_TASK_ID \
    --data-dir /home/snowm001/projects/OOD/Data/cade_lamda_4class_vt --prefix drebin_new --results-dir $WORK_DIR/results
