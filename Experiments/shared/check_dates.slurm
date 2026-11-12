#!/bin/bash
#SBATCH --job-name=dates
#SBATCH --partition=main
#SBATCH --output=logs/dates_%j.out
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --time=288:00:00
source /home/snowm001/projects/OOD/miniconda3/bin/activate OOD_val
cd /home/snowm001/projects/OOD
python3 - <<'PY'
import numpy as np, glob
for ds in ['Data2/Drebin/cade_drebin', 'Data2/EMBER/cade_ember2018_cadesplit_1000_k8', 'Data2/LAMDA/cade_lamda_old8_cadesplit_1000']:
    files = sorted(glob.glob(ds + '/**/*.npz', recursive=True))
    print('===', ds, len(files), 'npz files')
    if not files: continue
    d = np.load(files[0], allow_pickle=True)
    print('  keys of', files[0], {k: d[k].shape for k in d.files})
    for k in d.files:
        if any(w in k.lower() for w in ('date', 'time', 'month', 'year', 'first')):
            vals = np.concatenate([np.load(f, allow_pickle=True)[k].ravel() for f in files])
            v = sorted(map(str, vals))
            print(f'  {k}: from {v[0]} to {v[-1]}')
PY
