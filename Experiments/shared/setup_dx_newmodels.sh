#!/bin/bash
# setup_dx_newmodels.sh -- run from Experiments2 (bash):   bash setup_dx_newmodels.sh --dry   (check)   then   bash setup_dx_newmodels.sh
# Adds CADE-mask explanation runs (drift_analysis_cade) for the 7 models not yet explained:
#   QiCD pair only : classical_fidtwin_{c,p,r}_702_paironly   (copied from classical_fidtwin_*_702, model path -> ablation_objective)
#   CM pair+proto  : cade_{full,tuned}_8_702_proto (tau 10), cade_{enc,deep}_8_702_proto (tau 1)  (copied from cade_*_8_702)
# Same code, data, seed run 0, 200 inspected samples per split, 8 splits, on EMBER, Drebin, LAMDA.
# Then submits one job (after all of them) that aggregates and ranks all 16 models: rank_explainability16.py.
set -e
DRY=0; [ "$1" == "--dry" ] && DRY=1
NEW=()
make () {   # make <ds> <src_tag> <new_tag> <old_results_path> <new_results_path>
  local ds=$1 T=$2 N=$3 OLD=$4 NP=$5
  local src=$ds/pca8/drift_analysis_cade/$T dst=$ds/pca8/drift_analysis_cade/$N
  [ -d "$src" ] || { echo "!! missing $src"; exit 1; }
  [ -d "$ds/pca8/${NP#../../}/run0/split0" ] || { echo "!! trained model not found: $ds/pca8/${NP#../../}/run0/split0"; exit 1; }
  mkdir -p $dst/logs $dst/results
  for f in $src/*.py $src/*.slurm; do
    nf=$(basename $f | sed "s#$T#$N#")
    sed -e "s#'$OLD'#'$NP'#" \
        -e "s#^MODEL_TAG = '$T'#MODEL_TAG = '$N'#" \
        -e "s#run_drift_$T\.#run_drift_$N.#g" \
        -e "s#^\(\#SBATCH --job-name=.*\)\$#\1n#" \
        "$f" > $dst/$nf
  done
  grep -q "^DEFAULT_MODEL_RESULTS = '$NP'" $dst/run_drift_$N.py || { echo "!! model path not replaced in $dst"; exit 1; }
  grep -q "^MODEL_TAG = '$N'" $dst/run_drift_$N.py || { echo "!! MODEL_TAG not replaced in $dst"; exit 1; }
  echo "ok  $dst  ->  $NP"; NEW+=("$dst")
}
for ds in EMBER Drebin lamda; do
  for v in c p r; do
    T=classical_fidtwin_${v}_702
    make $ds $T ${T}_paironly "../../twin_pca8_p702/$T/results" "../../ablation_objective/${T}_paironly/results"
  done
  for vt in full:10 enc:1 tuned:10 deep:1; do
    v=${vt%%:*}; tau=${vt##*:}; T=cade_${v}_8_702
    make $ds $T ${T}_proto "../../cade_pca8_p702/$T/results" "../../cade_pca8_p702_proto/${T}_proto/tau_$tau/results"
  done
done
[ $DRY == 1 ] && { echo "dry run: ${#NEW[@]} folders created, nothing submitted"; exit 0; }
JOBS=()
for d in "${NEW[@]}"; do
  s=$(ls $d/*.slurm | head -1)
  j=$( cd $d && sbatch --parsable $(basename $s) ); JOBS+=($j); echo "$d: job $j"
done
dep=$(IFS=:; echo "${JOBS[*]}")
jf=$(sbatch --parsable --dependency=afterany:$dep finish_dx_newmodels.slurm)
echo "finish job $jf (runs after ${#JOBS[@]} array jobs)"
