#!/bin/bash
# PennyLane PCA-8 hybrid v7, 702 params (3 qubits, 62 layers, 24 cycling re-uploads, a = 0.5): 24 tasks = 8 held-out families x 3 seeds. Submit from this folder:
#   mkdir -p logs && sbatch run_pennylane_hybrid_v7_p702.slurm
#SBATCH --job-name=la_abl_qpo
#SBATCH --partition=main
#SBATCH --array=0-23
#SBATCH --output=logs/v7_p702_%A_%a.out
#SBATCH --error=logs/v7_p702_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=288:00:00
source /home/snowm001/projects/OOD/miniconda3/bin/activate OOD_val
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$SLURM_SUBMIT_DIR"
python3 quantum_hybrid_v7_p702_paironly.py $SLURM_ARRAY_TASK_ID --data-dir /home/snowm001/projects/OOD/Data2/LAMDA/cade_lamda_old8_cadesplit_1000
