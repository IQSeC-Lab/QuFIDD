#!/bin/bash
# After all jobs finish:   sbatch aggregate_all.slurm      -> each model's results/aggregate_summary.txt, collected in aggregate_all.txt
#SBATCH --job-name=dr_p8_agg
#SBATCH --partition=main
#SBATCH --output=aggregate_all_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=288:00:00
cd "$SLURM_SUBMIT_DIR"
source /home/snowm001/projects/OOD/miniconda3/bin/activate OOD_val
(cd quantum_pca8/pennylane/quantum_hybrid_v7_p702 && python3 pennylane_hybrid_v7_p702.py aggregate) > aggregate_all.txt 2>&1
source /home/snowm001/projects/OOD/miniconda3/bin/activate OOD_env
for d in twin_pca8_p702/classical_fidtwin_{c,p,r}_702 cade_pca8_p702/cade_{full,enc,tuned,deep}_8_702; do
  (cd $d && python3 $(ls run_*.py | head -1) aggregate) >> aggregate_all.txt 2>&1
done
(cd pca_only && python3 run_pca_only.py aggregate) >> aggregate_all.txt 2>&1
