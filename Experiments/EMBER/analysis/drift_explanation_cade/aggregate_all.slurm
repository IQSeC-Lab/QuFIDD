#!/bin/bash
#SBATCH --job-name=em_dx_agg
#SBATCH --partition=main
#SBATCH --output=compare_drift_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=288:00:00
# After all jobs finish:  sbatch aggregate_all.slurm   -> per-model aggregate_summary.txt, compare_drift.txt/.csv
source /home/snowm001/projects/OOD/miniconda3/bin/activate OOD_env
cd "$SLURM_SUBMIT_DIR"
for m in */; do m=${m%/}; [ -f "$m/run_drift_$m.py" ] && (cd "$m" && python3 "run_drift_$m.py" aggregate > /dev/null); done
python3 compare_drift.py
