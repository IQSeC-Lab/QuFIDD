#!/bin/bash
#SBATCH --job-name=em_p702_twinR
#SBATCH --partition=main
#SBATCH --array=0-23
#SBATCH --output=logs/test_%A_%a.out
#SBATCH --error=logs/test_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=288:00:00
# 24 test tasks = 8 held-out families x 3 seeds (42, 123, 456). Submit from this folder:
#   mkdir -p logs && sbatch run_classical_fidtwin_r_702.slurm        then: python3 run_classical_fidtwin_r_702.py aggregate
source /home/snowm001/projects/OOD/miniconda3/bin/activate OOD_env
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results
python3 run_classical_fidtwin_r_702.py $SLURM_ARRAY_TASK_ID \
    --data-dir /home/snowm001/projects/OOD/Data2/EMBER/cade_ember2018_cadesplit_1000_k8 --prefix drebin_new --results-dir results
