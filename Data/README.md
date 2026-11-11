# Data

The leave-one-family-out splits used in the QuFiDD paper. Each dataset folder holds eight files,
`drebin_new_k.npz` (k = 0..7), in the split format of CADE (Yang et al., USENIX Security 2021).
File `drebin_new_k.npz` withholds family *k* from training: the oldest 80% of the samples of the
seven known families form the training set, and the newest 20% form the test set together with
all samples of the withheld family (the drifting class). The file names follow CADE's naming for
all three datasets.

| k | EMBER-2018 | DREBIN | LAMDA |
|---|------------|--------|-------|
| 0 | Xtrat  | FakeInstaller | DroidKungFu |
| 1 | Zbot   | DroidKungFu   | Plankton    |
| 2 | Ramnit | Plankton      | BaseBridge  |
| 3 | Sality | GingerMaster  | SMSreg      |
| 4 | Wapomi | BaseBridge    | Airpush     |
| 5 | Lethic | Iconosys      | FakeInst    |
| 6 | Ursnif | Kmin          | Shedun      |
| 7 | Kovter | FakeDoc       | SMSsend     |

## Sources and construction

- **EMBER-2018** (Anderson and Roth, 2018): eight families from the EMBER 2018 training file,
  at most 1,000 samples per family, drawn in proportion to each family's monthly counts.
  Details per split are in `EMBER/split_summary.json`.
- **DREBIN** (Arp et al., NDSS 2014): CADE's released eight-family splits, used unchanged.
- **LAMDA** (Haque et al., ICLR 2026): eight families, at most 1,000 samples per family
  (BaseBridge has only 401 samples). Details are in `LAMDA/split_summary.json`, and the family
  labels are in `LAMDA/family_to_label.json`.

## Preprocessing

The stored features are already preprocessed per split, with every step fitted on the training
set only: a variance threshold of 0.003 on all three datasets, preceded on EMBER-2018 by min-max
scaling. EMBER-2018 test features can therefore fall outside [0, 1], and some test columns have
low variance. The projection onto the first eight principal components is not stored; the model
code fits it on each training set.

## File contents

Each file contains `X_train`, `y_train`, `X_test`, and `y_test`. The EMBER-2018 and LAMDA files
also store the withheld family (`heldout_family`), the family names (`families`), and the indices
of the kept features (`feature_index`); the LAMDA files also store the year and month of each
sample (`year_month_train`, `year_month_test`).

```python
import numpy as np
d = np.load("EMBER/drebin_new_0.npz", allow_pickle=True)
print(d.files, d["X_train"].shape, d["X_test"].shape)
```
