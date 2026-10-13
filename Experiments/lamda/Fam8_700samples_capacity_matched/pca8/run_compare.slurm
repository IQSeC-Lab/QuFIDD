#!/bin/bash
#SBATCH --job-name=compare
#SBATCH --partition=main
#SBATCH --output=logs/compare_%j.out
#SBATCH --error=logs/compare_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=288:00:00
# Aggregate every model folder in this experiment, then compare them all (paired tests vs quantum_hybrid_v7).
source /home/snowm001/projects/OOD/miniconda3/bin/activate OOD_env
cd $SLURM_SUBMIT_DIR
for m in cade_full_8 cade_deep_8_384 quantum_hybrid_v7 classical_fidtwin_c classical_fidtwin_p classical_fidtwin_r; do
  [ -d $m ] && (cd $m && python3 run_$m.py aggregate)
done
[ -d pca_only ] && (cd pca_only && python3 run_pca_only.py aggregate)
# include the matching full-capacity models if they exist (e.g. classical_full_capacity_models/Fam8_500samples)
N=$(basename $(dirname $SLURM_SUBMIT_DIR) | sed -n 's/Fam8_\([0-9]*\)samples.*/\1/p')
FULL=/home/snowm001/projects/OOD/Experiments/lamda/classical_full_capacity_models/Fam8_${N}samples
if [ -n "$N" ] && [ -d "$FULL" ]; then
  python3 compare_models.py --extra $FULL
else
  python3 compare_models.py
fi
