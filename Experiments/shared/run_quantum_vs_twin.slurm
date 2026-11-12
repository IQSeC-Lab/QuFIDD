#!/bin/bash
#SBATCH --job-name=q_vs_twin
#SBATCH --partition=main
#SBATCH --output=logs/q_vs_twin_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=288:00:00
# Submit from Experiments2:  mkdir -p logs && sbatch run_quantum_vs_twin.slurm
#   -> logs/q_vs_twin_<job>.out and QUANTUM_VS_TWIN_PVALUES.csv
source /home/snowm001/projects/OOD/miniconda3/bin/activate OOD_val
cd "$SLURM_SUBMIT_DIR"
python3 quantum_vs_twin_pvalues.py
