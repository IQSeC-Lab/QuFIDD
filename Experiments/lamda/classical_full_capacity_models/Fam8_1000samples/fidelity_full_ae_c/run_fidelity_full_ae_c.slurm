#!/bin/bash
#SBATCH --job-name=fidaeC_1000
#SBATCH --partition=main
#SBATCH --array=0-23
#SBATCH --output=logs/test_%A_%a.out
#SBATCH --error=logs/test_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=288:00:00
# 24 tasks = 8 held-out families x 3 seeds (42, 123, 456); full features, no PCA
source /home/snowm001/projects/OOD/miniconda3/bin/activate OOD_env
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
WORK_DIR=/home/snowm001/projects/OOD/Experiments/lamda/classical_full_capacity_models/Fam8_1000samples/fidelity_full_ae_c
cd $WORK_DIR
mkdir -p logs results
python3 run_fidelity_full_ae_c.py $SLURM_ARRAY_TASK_ID \
    --data-dir /home/snowm001/projects/OOD/Data/cade_lamda --prefix drebin_new --results-dir $WORK_DIR/results
