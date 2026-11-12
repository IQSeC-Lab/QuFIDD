#!/bin/bash
# Follow-up runs, started after collect_all has finished (dependency):
#  1. Drebin fixed-amplitude model: CADE-style drift analysis, then its quantum-native analysis
#  2. quantum-native analysis of the quantum models again, now with CADE's k on the same samples (E1)
# Usage (from ~/projects/OOD/Experiments2):   bash followups.sh <collect_all job id>
cd "$(dirname "$0")"; dep=${1:+--dependency=afterany:$1}
q=Drebin/pca8/drift_analysis_cade/quantum_amp_fixed_p702
j=$(cd $q && mkdir -p logs results && sbatch --parsable $dep run_drift_quantum_amp_fixed_p702.slurm); echo "Drebin amp CADE-style: $j"
(cd Drebin/pca8/drift_analysis_quantum/quantum_amp_fixed_p702 && mkdir -p logs results && sbatch --dependency=afterany:$j run_qexplain_quantum_amp_fixed_p702.slurm)
for D in EMBER Drebin; do
  for m in quantum_hybrid_v7_p702 quantum_amp_fixed_p702; do
    [ $D = Drebin ] && [ $m = quantum_amp_fixed_p702 ] && continue
    (cd $D/pca8/drift_analysis_quantum/$m && mkdir -p logs results && sbatch $dep run_qexplain_$m.slurm)
  done
done
