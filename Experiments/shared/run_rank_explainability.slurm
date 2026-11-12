#!/bin/bash
# Explainability ranking (CADE protocol) over EMBER and Drebin: all explained samples at FPR 10% and 20%, and drifting samples only.
# Submit from ~/projects/OOD/Experiments2:  sbatch run_rank_explainability.slurm
#SBATCH --job-name=rank_expl
#SBATCH --partition=main
#SBATCH --output=rank_explainability_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=288:00:00
source /home/snowm001/projects/OOD/miniconda3/bin/activate OOD_val
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$SLURM_SUBMIT_DIR"; mkdir -p explainability
python3 rank_explainability.py --fpr 0.10 --out explainability
python3 rank_explainability.py --fpr 0.20 --out explainability
python3 rank_explainability.py --fpr 0.10 --drift-only --out explainability
