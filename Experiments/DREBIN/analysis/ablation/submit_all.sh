#!/bin/bash
# Submit the five objective-ablation arrays (24 tasks each) of this dataset. Run in bash from anywhere.
cd "$(dirname "$0")"
for m in */; do m=${m%/}
  s=$(ls "$m"/*.slurm | grep -v aggregate | head -1)
  (cd "$m" && mkdir -p logs results && sbatch "$(basename "$s")")
done
