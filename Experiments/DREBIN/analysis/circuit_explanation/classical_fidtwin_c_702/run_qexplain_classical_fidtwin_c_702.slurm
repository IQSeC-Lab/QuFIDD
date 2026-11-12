#!/bin/bash
# Quantum-native drift explanation (re-upload decomposition, Pauli decomposition, prototype mixture + evaluations) for the fidelity twin, complex (M3 only)
# Drebin: one array task per split (8), seed run 0. Submit from this folder:
#   mkdir -p logs results && sbatch run_qexplain_classical_fidtwin_c_702.slurm        then: python3 run_qexplain_classical_fidtwin_c_702.py aggregate
#SBATCH --job-name=dr_qx_twc
#SBATCH --partition=main
#SBATCH --array=0-7
#SBATCH --output=logs/qexplain_%A_%a.out
#SBATCH --error=logs/qexplain_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=288:00:00
source /home/snowm001/projects/OOD/miniconda3/bin/activate OOD_env
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results
python3 run_qexplain_classical_fidtwin_c_702.py $SLURM_ARRAY_TASK_ID --data-dir /home/snowm001/projects/OOD/Data2/Drebin/cade_drebin --prefix drebin_new --run 0 --n-samples 200
