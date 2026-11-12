# QuFiDD: Quantum Fidelity-based Drift Detector

Code, configurations and results for **QuFiDD**, a variational quantum drift detector for malware concept drift. QuFiDD encodes each sample as a quantum state, represents each known family by a pure-state prototype, and scores a test sample by its fidelity distance to the closest prototype. It is compared against capacity-matched quantum-inspired classical detectors (QiCD) and CADE-style classical detectors (CM). Every model uses the same 8-dimensional PCA input and about 700 parameters.

## Repository layout

```
Data/
  EMBER/  DREBIN/  LAMDA/       leave-one-family-out splits used in the paper
Experiments/
  EMBER/  DREBIN/  LAMDA/       one folder per dataset, same structure in each
    models/                     training and evaluation code and results per model
    baselines/PCA_only/         drift score computed directly on the PCA features
    analysis/                   drift explanation, circuit inspection, FPR budgets,
                                threshold sweeps, objective ablation
    scripts/                    job submission, aggregation and sweep scripts
  shared/                       cross-dataset scripts: result collection, statistics
                                (seed std, p-values), plots and tables in the paper,
                                validation of our CADE explanation port
```

Model checkpoints are not included.

## Models

| Folder | Model | Description |
|---|---|---|
| `QuFiDD_pair`, `QuFiDD_pair+proto` | QuFiDD | 3-qubit variational circuit with data re-uploading, fidelity geometry |
| `QiCD-{c,p,r}_*` | QiCD | quantum-inspired classical detectors on complex (c), plain (p) or real (r) unit-norm states, fidelity geometry |
| `CM-f_r+pair*`, `CM-{e,t,d}_*` | CM | CADE-mini detectors (full, encoder, tuned, deep), Euclidean geometry |

Suffixes give the training objective: `pair` is CADE's pairwise contrastive loss, `pair+proto` adds a prototype loss, and `r` adds CADE's reconstruction loss.

## Protocol

- Datasets: EMBER-2018 (Windows), DREBIN and LAMDA (Android), each with 8 families.
- Each split withholds one family from training. Known-family samples are split by time, with the oldest 80% used for training.
- Preprocessing is fitted on the training set only: variance threshold 0.003 (plus min–max scaling on EMBER), then PCA to 8 components.
- Detection metrics: ROC-AUC, PR-AUC, CADE's ranked F1 and F1 under false-positive budgets from 10% to 60%.
- Drift explanation follows CADE's evaluation (Section 5 of the CADE paper).
- Every configuration is run with three seeds. Results are reported as mean ± std over seeds.

## Notes

- The quantum models are simulated with PennyLane. The classical models use PyTorch.
- The scripts were run on a SLURM cluster and still use their original absolute paths. They document exactly what was run, but the paths must be adapted before rerunning.
