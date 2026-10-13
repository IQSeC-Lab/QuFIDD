#!/bin/bash
#SBATCH --job-name=compare
#SBATCH --partition=main
#SBATCH --output=logs/compare_%j.out
#SBATCH --error=logs/compare_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=288:00:00
# Aggregate every model folder here (any folder with a run_<name>.py), then compare them all
# (paired tests vs the quantum model in this folder).
source /home/snowm001/projects/OOD/miniconda3/bin/activate OOD_env
cd $SLURM_SUBMIT_DIR
for d in */; do
  m=${d%/}
  [ -f $m/run_$m.py ] && (cd $m && python3 run_$m.py aggregate)
done
python3 compare_models.py
