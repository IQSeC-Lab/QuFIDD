#!/bin/bash
# After the array finishes:  bash tsne_summary.sh   -> one table of all splits + figures/tsne_ember_all.pdf (all panels on one sheet)
for f in $(ls tsne_all_*_*.out | sort -t_ -k4 -n); do
  fam=$(grep -o 'unseen family [^,]*' $f | cut -d' ' -f3)
  echo "== $fam"; grep -E "CHECK|ROC-AUC" $f | grep -v "OK$" | sed 's/drifting samples with mostly-known 10-NN/absorbed/'
done
python3 - <<'PY'
import glob, subprocess
fs = sorted(glob.glob('figures/fig_tsne_ember_*.pdf'))
try:
    subprocess.run(['pdfunite'] + fs + ['figures/tsne_ember_all.pdf'], check=True); print('Saved figures/tsne_ember_all.pdf (', len(fs), 'pages )')
except Exception as e:
    print('pdfunite not available; send the individual PDFs:', fs)
PY
