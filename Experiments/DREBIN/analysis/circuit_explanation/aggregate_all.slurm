#!/bin/bash
#SBATCH --job-name=dr_qx_agg
#SBATCH --partition=main
#SBATCH --output=compare_qexplain_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=288:00:00
# After all jobs finish:  sbatch aggregate_all.slurm   -> per-model aggregate_summary.txt, compare_qexplain.txt
source /home/snowm001/projects/OOD/miniconda3/bin/activate OOD_env
cd "$SLURM_SUBMIT_DIR"
for m in */; do m=${m%/}; [ -f "$m/run_qexplain_$m.py" ] && (cd "$m" && python3 "run_qexplain_$m.py" aggregate > /dev/null); done
python3 compare_qexplain.py
