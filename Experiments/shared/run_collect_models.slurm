#!/bin/bash
#SBATCH --job-name=collect_models
#SBATCH --partition=main
#SBATCH --output=collect_models_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=288:00:00
source /home/snowm001/projects/OOD/miniconda3/bin/activate OOD_val
cd "$SLURM_SUBMIT_DIR"
for d in EMBER Drebin lamda; do
  q=$d/pca8/quantum_pca8/pennylane
  for t in v8r v8e v8re; do [ -d $q/quantum_hybrid_${t}_p702 ] && (cd $q/quantum_hybrid_${t}_p702 && python3 quantum_hybrid_${t}_p702.py aggregate > /dev/null 2>&1); done
  [ -f $q/mw_entanglement.py ] && (cd $q && python3 mw_entanglement.py > /dev/null 2>&1)
  a=$d/pca8/ablation_objective
  [ -f $a/compare_ablation.py ] && (cd $a && python3 compare_ablation.py > /dev/null 2>&1)
done
python3 collect_models.py
echo; for d in EMBER Drebin lamda; do echo "=== $d entanglement"; cat $d/pca8/quantum_pca8/pennylane/mw_entanglement.txt 2>/dev/null; echo "=== $d objective ablation"; cat $d/pca8/ablation_objective/compare_ablation.txt 2>/dev/null; done
