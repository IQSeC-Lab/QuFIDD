#!/bin/bash
# Submit the drift-analysis job of every model in this folder whose trained results exist (run in bash).
cd "$(dirname "$0")"
for m in */; do m=${m%/}; [ -f "$m/run_drift_$m.slurm" ] || continue
  mr=$(grep "^DEFAULT_MODEL_RESULTS" "$m/run_drift_$m.py" | cut -d"'" -f2)
  if [ -d "$m/$mr/run0" ]; then (cd "$m" && mkdir -p logs results && sbatch "run_drift_$m.slurm"); else echo "skip $m (no trained results at $mr)"; fi
done
