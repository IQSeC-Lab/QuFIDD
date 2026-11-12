#!/bin/bash
# After the array finishes:  sbatch run_aggregate.slurm  -> aggregate_summary.txt, aggregate.csv, aggregate_plots/
#SBATCH --job-name=dr_abl_qpo_agg
#SBATCH --partition=main
#SBATCH --output=logs/aggregate_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=288:00:00
source /home/snowm001/projects/OOD/miniconda3/bin/activate OOD_val
cd "$SLURM_SUBMIT_DIR"
python3 quantum_hybrid_v7_p702_paironly.py aggregate
