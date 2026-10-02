#!/bin/bash
#SBATCH --job-name=FN_cms
#SBATCH --partition=main
#SBATCH --array=0-11
#SBATCH --output=logs/final_%A_%a.out
#SBATCH --error=logs/final_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=288:00:00

# Final 12 tasks with the variant chosen by select-summary (selection/selected_variant.json).
# A fixed variant instead:  sbatch --export=ALL,EXTRA="--variant N",OUT=results_vN run_cade_mini_select.slurm
EXTRA=${EXTRA:-}
OUT=${OUT:-results}
echo "cade_mini_select task $SLURM_ARRAY_TASK_ID  node $SLURMD_NODENAME  start $(date)"
source /home/snowm001/projects/OOD/miniconda3/bin/activate OOD_env
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
WORK_DIR=/home/snowm001/projects/OOD/Experiments/lamda/Full_new_experiments/cade_mini_select
cd $WORK_DIR
mkdir -p logs $OUT
python3 run_cade_mini_select.py $SLURM_ARRAY_TASK_ID $EXTRA \
    --data-dir /home/snowm001/projects/OOD/Data/cade_lamda_4class_vt --prefix drebin_new \
    --results-dir $WORK_DIR/$OUT
echo "end $(date)"
