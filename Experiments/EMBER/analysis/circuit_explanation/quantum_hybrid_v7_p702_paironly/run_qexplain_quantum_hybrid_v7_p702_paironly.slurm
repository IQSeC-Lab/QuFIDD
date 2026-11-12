#!/bin/bash
# Quantum-native drift explanation (re-upload decomposition, Pauli decomposition, prototype mixture + evaluations) for the quantum hybrid detector trained with CADE's objective (pair loss only) (PCA-8, 3 qubits, 62 layers, 24 cycling re-uploads, 702 params)
# EMBER: one array task per split (8), seed run 0. Submit from this folder:
#   mkdir -p logs results && sbatch run_qexplain_quantum_hybrid_v7_p702_paironly.slurm        then: python3 run_qexplain_quantum_hybrid_v7_p702_paironly.py aggregate
#SBATCH --job-name=em_qx_qp
#SBATCH --partition=main
#SBATCH --array=0-7
#SBATCH --output=logs/qexplain_%A_%a.out
#SBATCH --error=logs/qexplain_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=288:00:00
source /home/snowm001/projects/OOD/miniconda3/bin/activate OOD_val
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results
python3 run_qexplain_quantum_hybrid_v7_p702_paironly.py $SLURM_ARRAY_TASK_ID --data-dir /home/snowm001/projects/OOD/Data2/EMBER/cade_ember2018_cadesplit_1000_k8 --prefix drebin_new --run 0 --n-samples 200
