#!/bin/bash
#SBATCH --job-name=P256_cade_256
#SBATCH --partition=main
#SBATCH --array=0-3
#SBATCH --output=logs/cade_256_%A_%a.out
#SBATCH --error=logs/cade_256_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=288:00:00

# Pilot = seed 42, 4 families (tasks 0-3). Seeds 123/456 later: sbatch --array=4-11 run_cade_256.slurm
echo "pca256_experiment/cade_256 task $SLURM_ARRAY_TASK_ID  node $SLURMD_NODENAME  start $(date)"
source /home/snowm001/projects/OOD/miniconda3/bin/activate OOD_env
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
WORK_DIR=/home/snowm001/projects/OOD/Experiments/lamda/pca256_experiment/cade_256
cd $WORK_DIR
mkdir -p logs results
python3 run_cade_256.py $SLURM_ARRAY_TASK_ID \
    --data-dir /home/snowm001/projects/OOD/Data/cade_lamda_4class_vt --prefix drebin_new \
    --results-dir $WORK_DIR/results
echo "end $(date)"
