#!/bin/bash
#SBATCH --job-name=dx_finish
#SBATCH --partition=main
#SBATCH --output=dx_finish_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=288:00:00
# Submit from Experiments2 when all drift_analysis jobs are done:  sbatch finish_drift_explanation.slurm
# 1) aggregate every new folder  2) explanation ranking over EMBER, Drebin, LAMDA (incl. the CADE-objective quantum detector)
# 3) collect the data for the drift-explanation section
source /home/snowm001/projects/OOD/miniconda3/bin/activate OOD_val
cd "$SLURM_SUBMIT_DIR"
for d in lamda/pca8/drift_analysis_cade/*/ lamda/pca8/drift_analysis_quantum/*/ */pca8/drift_analysis_cade/quantum_hybrid_v7_p702_paironly/ */pca8/drift_analysis_quantum/quantum_hybrid_v7_p702_paironly/; do
  s=$(ls $d/run_*.py 2>/dev/null | head -1); [ -n "$s" ] || continue
  echo "== aggregate $d"; ( cd $d && python3 $(basename $s) aggregate | tail -3 )
done
python3 rank_explainability.py --datasets EMBER Drebin lamda --fpr 0.10 --drift-only
python3 collect_drift_explanation.py
