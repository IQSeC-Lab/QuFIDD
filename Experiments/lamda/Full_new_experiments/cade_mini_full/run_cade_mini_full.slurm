#!/bin/bash
#SBATCH --job-name=FN_cade_mini_full
#SBATCH --partition=main
#SBATCH --array=0-11
#SBATCH --output=logs/cade_mini_full_%A_%a.out
#SBATCH --error=logs/cade_mini_full_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=288:00:00

# All 12 tasks (3 seeds x 4 families):   sbatch run_cade_mini_full.slurm
# Extra args / other results folder:     sbatch --export=ALL,EXTRA="--max-epochs 0",OUT=results_untrained run_cade_mini_full.slurm
EXTRA=${EXTRA:-}
OUT=${OUT:-results}

echo "Full_new_experiments/cade_mini_full  task $SLURM_ARRAY_TASK_ID  job $SLURM_JOB_ID  node $SLURMD_NODENAME  start $(date)"
source /home/snowm001/projects/OOD/miniconda3/bin/activate OOD_env
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

WORK_DIR=/home/snowm001/projects/OOD/Experiments/lamda/Full_new_experiments/cade_mini_full
cd $WORK_DIR
mkdir -p logs $OUT
python3 run_cade_mini_full.py $SLURM_ARRAY_TASK_ID $EXTRA \
    --data-dir /home/snowm001/projects/OOD/Data/cade_lamda_4class_vt --prefix drebin_new \
    --results-dir $WORK_DIR/$OUT
echo "end $(date)"
