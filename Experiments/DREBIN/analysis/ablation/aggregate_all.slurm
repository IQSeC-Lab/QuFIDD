#!/bin/bash
#SBATCH --job-name=dr_abl_agg
#SBATCH --partition=main
#SBATCH --output=compare_ablation_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=288:00:00
# After all ablation arrays finish:  sbatch aggregate_all.slurm  -> per-variant aggregate_summary.txt + compare_ablation.txt/.csv
source /home/snowm001/projects/OOD/miniconda3/bin/activate OOD_val
cd "$SLURM_SUBMIT_DIR"
for m in */; do m=${m%/}; p=$(ls "$m"/*.py | head -1); (cd "$m" && python3 "$(basename "$p")" aggregate > /dev/null); done
python3 compare_ablation.py
