#!/bin/bash
#SBATCH --job-name=compare_cade
#SBATCH --partition=main
#SBATCH --output=logs/compare_vs_cade_%j.out
#SBATCH --error=logs/compare_vs_cade_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=288:00:00
# Same comparison as run_compare.slurm, but the paired Wilcoxon tests use CADE original (full features) as the
# reference. Writes comparison_vs_cade.txt / .csv; comparison.txt (tests vs v7) is left untouched.
source /home/snowm001/projects/OOD/miniconda3/bin/activate OOD_env
cd $SLURM_SUBMIT_DIR
N=$(basename $(dirname $SLURM_SUBMIT_DIR) | sed -n 's/Fam8_\([0-9]*\)samples.*/\1/p')
[ "$(basename $(dirname $SLURM_SUBMIT_DIR))" = "capacity_matched_8fam" ] && N=1000
FULL=/home/snowm001/projects/OOD/Experiments/lamda/classical_full_capacity_models/Fam8_${N}samples
python3 compare_models.py --extra $FULL --ref cade_original --tag vs_cade
