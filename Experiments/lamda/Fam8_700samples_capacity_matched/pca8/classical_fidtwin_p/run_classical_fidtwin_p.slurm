#!/bin/bash
#SBATCH --job-name=p8_twinP_700
#SBATCH --partition=main
#SBATCH --array=0-23
#SBATCH --output=logs/test_%A_%a.out
#SBATCH --error=logs/test_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=288:00:00
# 24 test tasks = 8 held-out families x 3 seeds (42, 123, 456)
source /home/snowm001/projects/OOD/miniconda3/bin/activate OOD_env
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
WORK_DIR=/home/snowm001/projects/OOD/Experiments/lamda/Fam8_700samples_capacity_matched/pca8/classical_fidtwin_p
cd $WORK_DIR
mkdir -p logs results validation
python3 run_classical_fidtwin_p.py $SLURM_ARRAY_TASK_ID \
    --data-dir /home/snowm001/projects/OOD/Data/cade_lamda_700_vt --prefix drebin_new --results-dir $WORK_DIR/results
