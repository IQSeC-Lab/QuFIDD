#!/bin/bash
# setup_drift_lamda_paironly.sh -- run from Experiments2 (bash):   bash setup_drift_lamda_paironly.sh [--dry]
set -e
DRY=0; [ "$1" == "--dry" ] && DRY=1
if [ -n "$LAMDA_DATA_OVERRIDE" ]; then LAMDA_DATA=$LAMDA_DATA_OVERRIDE; else LAMDA_DATA=$(/home/snowm001/projects/OOD/miniconda3/envs/OOD_val/bin/python - <<'EOF'
import os, importlib.util
d = 'lamda/pca8/cade_pca8_p702/cade_enc_8_702'
s = [f for f in os.listdir(d) if f.startswith('run_') and f.endswith('.py')][0]
spec = importlib.util.spec_from_file_location('m', os.path.join(d, s)); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print(os.path.expanduser(m.DEFAULT_DATA))
EOF
); fi
DREBIN_DATA=/home/snowm001/projects/OOD/Data2/Drebin/cade_drebin
echo "LAMDA data: $LAMDA_DATA"
[ -d "$LAMDA_DATA" ] || { echo "LAMDA data dir not found"; exit 1; }
for A in drift_analysis_cade drift_analysis_quantum; do
  for src in Drebin/pca8/$A/*/; do
    m=$(basename $src); [ "$m" == "feature_names" ] && continue
    dst=lamda/pca8/$A/$m; mkdir -p $dst/logs $dst/results
    for f in $src/*.py $src/*.slurm; do
      [ -f "$f" ] || continue
      sed -e "s#$DREBIN_DATA#$LAMDA_DATA#g" \
          -e "s#^DATASET = 'Drebin'#DATASET = 'lamda'#" \
          -e "s#^BINARY = DATASET == 'Drebin'#BINARY = True   \# LAMDA and Drebin features are binary#" \
          -e "s#--job-name=dr_#--job-name=la_#" \
          -e "s#^\# Drebin: #\# LAMDA: #" \
          -e "s# --feature-names-dir ../feature_names##" \
          "$f" > $dst/$(basename $f)
    done
  done
done
echo "created lamda/pca8/drift_analysis_{cade,quantum}/: $(ls lamda/pca8/drift_analysis_cade | tr '\n' ' ')"
for ds in EMBER Drebin lamda; do
  for A in drift_analysis_cade drift_analysis_quantum; do
    src=$ds/pca8/$A/quantum_hybrid_v7_p702; dst=$ds/pca8/$A/quantum_hybrid_v7_p702_paironly
    [ -d "$src" ] || { echo "missing $src"; continue; }
    mkdir -p $dst/logs $dst/results
    for f in $src/*.py $src/*.slurm; do
      [ -f "$f" ] || continue
      nf=$(basename $f | sed 's#quantum_hybrid_v7_p702#quantum_hybrid_v7_p702_paironly#')
      sed -e "s#'../../quantum_pca8/pennylane/quantum_hybrid_v7_p702/results'#'../../ablation_objective/quantum_hybrid_v7_p702_paironly/results'#" \
          -e "s#drift_analysis_cade/quantum_hybrid_v7_p702/results#drift_analysis_cade/quantum_hybrid_v7_p702_paironly/results#" \
          -e "s#^MODEL_TAG = 'quantum_hybrid_v7_p702'#MODEL_TAG = 'quantum_hybrid_v7_p702_paironly'#" \
          -e "s#run_drift_quantum_hybrid_v7_p702\.#run_drift_quantum_hybrid_v7_p702_paironly.#g" \
          -e "s#run_qexplain_quantum_hybrid_v7_p702\.#run_qexplain_quantum_hybrid_v7_p702_paironly.#g" \
          -e "s#^\(\#SBATCH --job-name=[a-z]*_[a-z]*_q\)\$#\1p#" \
          -e "s#for the quantum hybrid detector#for the quantum hybrid detector trained with CADE's objective (pair loss only)#" \
          "$f" > $dst/$nf
    done
    grep -q "ablation_objective/quantum_hybrid_v7_p702_paironly" $dst/*.py || { echo "!! model path not replaced in $dst"; exit 1; }
  done
done
echo "created quantum_hybrid_v7_p702_paironly folders"
[ $DRY == 1 ] && { echo "dry run: nothing submitted"; exit 0; }
submit () {
  local d=$1 dep=$2 s
  s=$(ls $d/*.slurm | head -1)
  ( cd $d && mkdir -p logs results && if [ -n "$dep" ]; then sbatch --parsable --dependency=afterok:$dep $(basename $s); else sbatch --parsable $(basename $s); fi )
}
for ds in EMBER Drebin lamda; do
  if [ $ds == lamda ]; then models=$(ls lamda/pca8/drift_analysis_cade); else models=quantum_hybrid_v7_p702_paironly; fi
  for m in $models; do
    j=$(submit $ds/pca8/drift_analysis_cade/$m)
    echo "$ds $m: drift_analysis_cade job $j"
    if [ -d $ds/pca8/drift_analysis_quantum/$m ]; then
      jq=$(submit $ds/pca8/drift_analysis_quantum/$m $j)
      echo "$ds $m: drift_analysis_quantum job $jq (after $j)"
    fi
  done
done
