#!/bin/bash
source /home/snowm001/projects/OOD/miniconda3/bin/activate OOD_env
cd ~/projects/OOD/Experiments/lamda/capacity_matched_8fam
for P in pca8 pca64; do
  echo "################ $P: evaluate"
  python3 evaluate.py $P
  echo "################ $P: compare"
  (cd $P && python3 compare_all.py --n-boot 500 > comparison/compare_all_full.log 2>&1)
  echo "################ $P: latent spaces"
  python3 plot_latent_spaces.py $P
done
