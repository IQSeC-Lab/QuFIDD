# QuFiDD: Quantum Fidelity-based Drift Detector

Code, configurations and results for **QuFiDD**, a variational quantum drift detector for malware concept drift. QuFiDD encodes each sample as a quantum state, represents each known family by a pure-state prototype, and scores a test sample by its fidelity distance to the closest prototype. It is compared against capacity-matched quantum-inspired classical detectors (QiCD) and CADE-style classical detectors (CM). Every model uses the same 8-dimensional PCA input and about 700 parameters.

## Repository layout

```
Data/
  EMBER/  LAMDA/                leave-one-family-out splits used in the paper
  DREBIN/                       not included, see "Data" below
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
LICENSE
requirements.txt
```

Model checkpoints are not included.

## Data

- **EMBER-2018** and **LAMDA**: the leave-one-family-out splits are in `Data/EMBER/` and `Data/LAMDA/`.
- **DREBIN**: we use the preprocessed DREBIN splits released with CADE (Yang et al., USENIX Security 2021). They are not redistributed here. Download `drebin_new_0.npz` to `drebin_new_7.npz` from the `data` folder of the CADE repository, https://github.com/whyisyoung/CADE, and place them in `Data/DREBIN/`.

## Installation

The experiments were run with Python 3.11. Install the packages from `requirements.txt` in either a conda environment or a Python virtual environment.

**Option 1: conda**

```bash
conda create -n qufidd python=3.11
conda activate qufidd
pip install -r requirements.txt
```

**Option 2: Python venv**

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

The quantum models are simulated with PennyLane. The classical models use PyTorch. A GPU is not required. For a CUDA build of PyTorch, follow the instructions at https://pytorch.org before running `pip install -r requirements.txt`.

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

The scripts were run on a SLURM cluster and still use their original absolute paths. They document exactly what was run, but the paths must be adapted before rerunning.

## Acknowledgments and citations

The DREBIN splits, the CADE baseline and the drift-explanation protocol come from CADE. If you use the data in this repository, please cite the original dataset papers and CADE:

```bibtex
@inproceedings{yang2021cade,
  title     = {{CADE}: Detecting and Explaining Concept Drift Samples for Security Applications},
  author    = {Yang, Limin and Guo, Wenbo and Hao, Qingying and Ciptadi, Arridhana and Ahmadzadeh, Ali and Xing, Xinyu and Wang, Gang},
  booktitle = {Proc. of USENIX Security},
  year      = {2021}
}

@inproceedings{arp2014drebin,
  title     = {{DREBIN}: Effective and Explainable Detection of Android Malware in Your Pocket},
  author    = {Arp, Daniel and Spreitzenbarth, Michael and Hubner, Malte and Gascon, Hugo and Rieck, Konrad},
  booktitle = {Proc. of NDSS},
  year      = {2014}
}

@article{anderson2018ember,
  title   = {{EMBER}: An Open Dataset for Training Static {PE} Malware Machine Learning Models},
  author  = {Anderson, Hyrum S. and Roth, Phil},
  journal = {arXiv preprint arXiv:1804.04637},
  year    = {2018}
}

@inproceedings{haque2026lamda,
  title     = {{LAMDA}: A Longitudinal {Android} Malware Benchmark for Concept Drift Analysis},
  author    = {Haque, Md Ahsanul and Hossain, Ismail and Kamol, Mahmuduzzaman and Alam, Jahangir and Amalapuram, Suresh Kumar and Talukder, Sajedul and Rahman, Mohammad Saidur},
  booktitle = {Proc. of ICLR},
  year      = {2026}
}
```

## License

This project is released under the MIT License. See `LICENSE`.
