#!/bin/bash
# Aggregate and collect ALL results (EMBER, Drebin, LAMDA) into Experiments2/ALL_RESULTS.txt
# Submit from ~/projects/OOD/Experiments2:   sbatch collect_all.slurm
#SBATCH --job-name=collect_all
#SBATCH --partition=main
#SBATCH --output=collect_all_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=288:00:00
cd "$SLURM_SUBMIT_DIR"
ACT=/home/snowm001/projects/OOD/miniconda3/bin/activate
OUT=ALL_RESULTS.txt; : > $OUT
sec() { printf '\n\n################################ %s\n' "$*" >> $OUT; }
# FPR-budget scripts that include the fixed-amplitude model (EMBER/Drebin copies from amppkg; LAMDA was copied already)
for D in EMBER Drebin; do [ -f amppkg/$D/pca8/fpr_ranked_f1_pca8.py ] && cp amppkg/$D/pca8/fpr_ranked_f1_pca8.py $D/pca8/; done

for D in EMBER Drebin lamda; do
  P=$D/pca8
  # ---------------- completeness check
  sec "$D: completeness (trained tasks per model, expected 24)"
  for d in $P/quantum_pca8/pennylane/* $P/twin_pca8_p702/* $P/cade_pca8_p702/*; do
    [ -d "$d" ] || continue
    printf '  %-45s %3s tasks\n' "${d#$P/}" "$(ls $d/results/run*/split*/summary.json 2>/dev/null | wc -l)" >> $OUT
  done
  for A in drift_analysis_cade drift_analysis_quantum; do
    [ -d $P/$A ] || continue
    echo "  $A (expected 8 splits each):" >> $OUT
    for m in $P/$A/*/; do m=${m%/}; ls $m/run_*.py >/dev/null 2>&1 || continue
      n=$(ls $m/results/split*/summary.json 2>/dev/null | wc -l)
      err=$(grep -l -E "Traceback|Error" $m/logs/*.err 2>/dev/null | wc -l)
      printf '    %-28s %2s splits  %s\n' "$(basename $m)" "$n" "$([ $err -gt 0 ] && echo "($err log files with errors)")" >> $OUT
    done
  done
  # ---------------- model aggregates
  sec "$D: model results (raw score, no filter)"
  source $ACT OOD_val
  for m in quantum_hybrid_v7_p702 quantum_amp_fixed_p702; do
    f=$P/quantum_pca8/pennylane/$m; [ -d $f/results ] && (cd $f && python3 $(ls *.py | head -1) aggregate) >> $OUT 2>&1
  done
  source $ACT OOD_env
  for d in $P/twin_pca8_p702/classical_fidtwin_{c,p,r}_702 $P/cade_pca8_p702/cade_{full,enc,tuned,deep}_8_702; do
    [ -d $d/results ] && (cd $d && python3 $(ls run_*.py | head -1) aggregate) >> $OUT 2>&1
  done
  [ -d $P/pca_only ] && (cd $P/pca_only && python3 run_pca_only.py aggregate) >> $OUT 2>&1
  # ---------------- threshold sweeps and FPR budgets (now including the fixed-amplitude model)
  sec "$D: threshold sweeps (training-percentile thresholds) and FPR budgets"
  (cd $P && mkdir -p logs && python3 threshold_sweep_pca8.py && python3 fpr_ranked_f1_pca8.py) >> $OUT 2>&1
  # ---------------- drift explanation analyses
  for A in drift_analysis_cade drift_analysis_quantum; do
    [ -d $P/$A ] || continue
    sec "$D: $A"
    (cd $P/$A && for m in */; do m=${m%/}; s=$(ls $m/run_*.py 2>/dev/null | head -1); [ -n "$s" ] && [ -d $m/results ] && (cd $m && python3 $(basename $s) aggregate > /dev/null 2>&1); done
     python3 $(ls compare_*.py | head -1)) >> $OUT 2>&1
    if [ $A = drift_analysis_quantum ]; then
      echo "  E1 used CADE's k (true) or the fallback k = 10 (false), per split:" >> $OUT
      grep -h '"k_from_cade"' $P/$A/quantum_*/results/split*/summary.json 2>/dev/null | sort | uniq -c >> $OUT
    fi
  done
done
sec "error logs (last lines of up to 6 failing tasks)"
for f in $(grep -l -E "Traceback" */pca8/drift_analysis_*/*/logs/*.err */pca8/quantum_pca8/pennylane/*/logs/*.err 2>/dev/null | head -6); do
  echo "--- $f" >> $OUT; tail -4 $f >> $OUT
done
echo "DONE $(date)" >> $OUT
