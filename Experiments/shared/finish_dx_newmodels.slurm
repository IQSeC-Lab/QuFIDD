#!/bin/bash
#SBATCH --job-name=dx_fin16
#SBATCH --partition=main
#SBATCH --output=dx_fin16_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=288:00:00
# Submitted by setup_dx_newmodels.sh after all explanation jobs. Aggregates the new folders and ranks all 16 models.
source /home/snowm001/projects/OOD/miniconda3/bin/activate OOD_val
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$SLURM_SUBMIT_DIR"
for d in */pca8/drift_analysis_cade/*_paironly/ */pca8/drift_analysis_cade/*_proto/; do
  s=$(ls $d/run_drift_*.py 2>/dev/null | head -1); [ -n "$s" ] || continue
  echo "== aggregate $d"; ( cd $d && python3 $(basename $s) aggregate | tail -2 )
done
echo; echo "== splits explained per new model (expect 8)"
for d in */pca8/drift_analysis_cade/classical_fidtwin_*_paironly/ */pca8/drift_analysis_cade/cade_*_proto/; do
  echo "$d $(ls $d/results/split*/explanations.csv 2>/dev/null | wc -l)"; done
mkdir -p explainability
python3 rank_explainability16.py --datasets EMBER Drebin lamda --fpr 0.10 --drift-only --out explainability
