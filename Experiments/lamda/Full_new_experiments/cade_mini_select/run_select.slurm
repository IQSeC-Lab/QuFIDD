#!/bin/bash
#SBATCH --job-name=FN_cms_sel
#SBATCH --partition=main
#SBATCH --array=0-215
#SBATCH --output=logs/select_%A_%a.out
#SBATCH --error=logs/select_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=288:00:00
source /home/snowm001/projects/OOD/miniconda3/bin/activate OOD_env
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd /home/snowm001/projects/OOD/Experiments/lamda/Full_new_experiments/cade_mini_select
mkdir -p logs selection
python3 run_cade_mini_select.py select --job $SLURM_ARRAY_TASK_ID \
    --data-dir /home/snowm001/projects/OOD/Data/cade_lamda_4class_vt --prefix drebin_new
