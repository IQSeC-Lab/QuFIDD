#!/bin/bash
#SBATCH --job-name=coll_ptau
#SBATCH --partition=main
#SBATCH --output=logs/coll_ptau_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=288:00:00
# Submit from Experiments2 after all tau arrays finish:  mkdir -p logs && sbatch run_collect_cade_proto_tau.slurm
#   1. runs each script's own "aggregate" step (per-family summaries, as for the original CADE-minis)
#   2. computes all Table I metrics for every run -> log, CADE_PROTO_TAU_ALL.csv, CADE_PROTO_BEST.csv
cd "$SLURM_SUBMIT_DIR"

source /home/snowm001/projects/OOD/miniconda3/bin/activate OOD_env
for d in */pca8/cade_pca8_p702_proto/*_proto/tau_*/; do
    py=$(ls "$d"run_*_proto.py 2>/dev/null | head -1)
    [ -n "$py" ] || continue
    echo "== aggregate $d"
    (cd "$d" && python3 "$(basename "$py")" aggregate > aggregate_stdout.txt 2>&1 || echo "   aggregate failed, see $d/aggregate_stdout.txt")
done

source /home/snowm001/projects/OOD/miniconda3/bin/activate OOD_val
python3 collect_cade_proto_tau.py
