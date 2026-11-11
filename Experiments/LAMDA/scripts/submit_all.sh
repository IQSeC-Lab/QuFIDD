#!/bin/bash
# Submit every PCA-8 model of this folder (LAMDA, old 8 families, capped 1000, CADE per-split time cut).
# Run from anywhere, in bash:   bash submit_all.sh
cd "$(dirname "$0")"
for d in quantum_pca8/pennylane/quantum_hybrid_v7_p702 twin_pca8_p702/classical_fidtwin_{c,p,r}_702 cade_pca8_p702/cade_{full,enc,tuned,deep}_8_702 pca_only; do
  s=$(ls $d/run_*.slurm | grep -v aggregate | head -1)
  (cd $d && mkdir -p logs results && sbatch $(basename $s))
done
