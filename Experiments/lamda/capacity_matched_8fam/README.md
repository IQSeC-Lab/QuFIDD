# capacity_matched_8fam — 8-family LAMDA, PCA-8 (angle) and PCA-64 (amplitude)

Two self-contained experiments on **8-family LAMDA** (`~/projects/OOD/Data/cade_lamda/drebin_new_{0..7}.npz`):
8 held-out families × 3 seeds = **24 tasks** each, with **7 known families** per task.
The CADE-mini bottleneck is therefore **7-dimensional** (CADE: latent size = number of known families), so the hidden
widths are re-chosen to stay as close as possible to the quantum parameter counts.
The flag-all F1 floor differs per split here (test sets differ in size), so it is computed from each task's test set.
Each has three CADE-mini versions (the PCA-5 designs, rescaled), one quantum model, and zero-parameter PCA baselines.
Every script is standalone; all models save per-task plots, aggregate plots, `ood_scores.csv`, and training scores.

## Models and parameter counts

| Experiment | Folder | Model | Params | 4-family version |
|---|---|---|---|---|
| **pca8** | `quantum_angle_8` | Quantum v2 design, angle encoding, 8 qubits, 8 layers, 4 uploads | **192** | 192 |
| | `cade_full_8` | encoder 8-6-7 + mirrored decoder, 0.1·contrastive (margin 10) + MSE | 207 | 8-8-3 (203) |
| | `cade_enc_8` | encoder 8-12-7, contrastive only, margin 10 | 199 | 8-16-3 (195) |
| | `cade_tuned_8` | encoder 8-12-7, contrastive only, margin 20 | 199 | 8-16-3 (195) |
| **pca64** | `quantum_amp_64` | amplitude encoding (64 amplitudes), 6 qubits, 25 layers, 2 angle re-uploads, pure prototypes | **474** | 474 |
| | `cade_full_64` | encoder 64-3-7 + mirrored decoder, 0.1·contrastive (margin 10) + MSE | 503 | 64-3-3 (475) |
| | `cade_enc_64` | encoder 64-6-7, contrastive only, margin 10 | 439 | 64-7-3 (479) |
| | `cade_tuned_64` | encoder 64-6-7, contrastive only, margin 20 | 439 | 64-7-3 (479) |

With a 7-dimensional bottleneck, one hidden unit costs 16 (PCA-8) or 72 (PCA-64) parameters per layer, so the PCA-64
CADE-minis bracket the quantum model (439 and 503 vs 474) rather than matching it to within a few parameters.

Plus `pca_only` in each experiment (L2-normalized and raw-coordinate baselines, 0 params).
The tuned design (encoder-only, margin 20, LR 0.001) is the one selected on validation at PCA-5; it is not reselected.

## Workflow (per experiment; example: pca8)

```bash
cd ~/projects/OOD/Experiments/lamda/capacity_matched_8fam
bash; source ~/projects/OOD/miniconda3/bin/activate OOD_env

# 1. quantum pre-check (no training)
(cd pca8/quantum_angle_8 && python3 run_quantum_angle_8.py probe)

# 2. test runs: 24 tasks per model (no validation runs are used in this study)
(cd pca8/pca_only && python3 run_pca_only.py all)
for m in cade_full_8 cade_enc_8 cade_tuned_8 quantum_angle_8; do (cd pca8/$m && mkdir -p logs && sbatch run_$m.slurm); done

# 3. after the jobs finish
for m in cade_full_8 cade_enc_8 cade_tuned_8 quantum_angle_8; do (cd pca8/$m && python3 run_$m.py aggregate); done
python3 evaluate.py pca8                                 # thresholded results (the agreed protocol)
(cd pca8 && python3 compare_all.py --n-boot 500)         # overall comparison + statistics + plots
```

For **pca64**, replace `pca8` with `pca64`, the model folders with `cade_full_64 cade_enc_64 cade_tuned_64 quantum_amp_64`,
and the quantum script with `run_quantum_amp_64.py`.

## Evaluation protocol (`evaluate.py`)

| Models | Score | Thresholds | Results |
|---|---|---|---|
| Classical + baselines | MAD score | T = 0, 2, 2.5, 3, 3.5 | ranked (CADE protocol) |
| Quantum | raw distance | observed range / 7 (k = 0 … 6; k = 0 = flag all) | flag-rule **and** ranked |
| Quantum | MAD score | observed range / 7 | flag-rule **and** ranked |
| Quantum | MAD score | T = 0, 1, 1.5, 2, 2.5, 3, 3.5 | flag-rule **and** ranked |

- **flag-rule:** every sample above T counts as detected (flagged share, P, R, F1, FNR, FPR).
- **ranked (CADE protocol):** flagged samples ranked by raw distance, best-F1 cut-off (P, R, F1, FNR, FPR, Effort).
- **No validation-based threshold selection:** every threshold of every scheme is reported for every model.
  (`select_threshold.py` and the quantum `validate` mode are kept but unused; `evaluate.py` only marks a `*` main row
  if a `validation/selected_thresholds.json` exists.)

Custom runs (any model, score, thresholds, mode):

```bash
python3 evaluate.py pca64 --models quantum --score mad --relative 15 --mode both --tag q_mad15
python3 evaluate.py pca8 --models cade --score mad --thresholds 0 1 2 3 3.5 4 --mode flag --tag cade_flag
python3 evaluate.py pca64 --by-family
```

Outputs: `<experiment>/evaluation/<tag>.txt` and `.csv` (default tag `protocol`).

## Model scripts (all)

```
python3 run_<model>.py <task 0-23>          # one test task
python3 run_<model>.py all                  # all 24 sequentially (baselines, quick tests)
python3 run_<model>.py aggregate            # aggregate table + plots
python3 run_<model>.py <task> --max-epochs N --results-dir DIR --data-dir DIR
quantum only:  probe
```
