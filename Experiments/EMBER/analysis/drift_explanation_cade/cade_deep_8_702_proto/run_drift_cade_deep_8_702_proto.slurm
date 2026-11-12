#!/bin/bash
# CADE Sec. 5 drift explanation analysis (random, boundary-based, COIN, CADE; port of the released code) for the CADE-mini, deep (8-23-16-7)
# EMBER: one array task per split (8 splits), seed run 0, 200 inspected samples per split (--n-samples 0 = all). Submit from this folder:
#   mkdir -p logs results && sbatch run_drift_cade_deep_8_702_proto.slurm        then: python3 run_drift_cade_deep_8_702_proto.py aggregate
#SBATCH --job-name=em_dx_cdeepn
#SBATCH --partition=main
#SBATCH --array=0-7
#SBATCH --output=logs/drift_%A_%a.out
#SBATCH --error=logs/drift_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=288:00:00
source /home/snowm001/projects/OOD/miniconda3/bin/activate OOD_env
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results
python3 run_drift_cade_deep_8_702_proto.py $SLURM_ARRAY_TASK_ID --data-dir /home/snowm001/projects/OOD/Data2/EMBER/cade_ember2018_cadesplit_1000_k8 --prefix drebin_new --run 0 --n-samples 200 --feature-names-dir ../feature_names
