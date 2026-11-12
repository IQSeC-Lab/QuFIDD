#!/bin/bash
# Submit every model of this folder whose trained results exist (run in bash).
cd "$(dirname "$0")"
for m in */; do m=${m%/}; [ -f "$m/run_qexplain_$m.slurm" ] || continue
  mr=$(grep "^DEFAULT_MODEL_RESULTS" "$m/run_qexplain_$m.py" | cut -d"'" -f2)
  if [ -d "$m/$mr/run0" ]; then (cd "$m" && mkdir -p logs results && sbatch "run_qexplain_$m.slurm"); else echo "skip $m (no trained results at $mr)"; fi
done
