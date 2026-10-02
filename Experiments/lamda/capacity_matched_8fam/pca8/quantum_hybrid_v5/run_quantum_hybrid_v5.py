#!/usr/bin/env python3
"""
run_quantum_hybrid_v5.py -- pca8 / quantum_hybrid_v5   (stacked ensemble of angle-8 + amp-8)
======================================================================================================================
Quantum hybrid v5 -- a LEARNED, sample-dependent combination of the two trained quantum encodings.
  members     the existing angle-8 and amp-8 models of the SAME seed and split (192 + 192 params; not retrained)
  features    per sample, from each member's raw distance standardized by that member's own TRAINING raw scores
              (z = (d - median) / (1.4826 MAD), signed):  [z_angle, z_amp, z_angle * z_amp, |z_angle - z_amp|]
              (the product and the absolute difference let the weighting depend on the sample: agreement vs conflict)
  combiner    logistic regression (4 weights + bias = 5 params) on standardized features, class-balanced,
              trained on VALIDATION folds only: in split s, each of 3 known families is held out of the training data
              as pseudo-drift; angle-8 and amp-8 are trained on the rest (seed 42) and score the held-out samples.
              ONE combiner PER SPLIT, from that split's folds only -- pooling across splits would leak, because the
              family held out as test drift in split s is a known family (and can be a pseudo-drift fold) in another.
  score       raw = combiner logit; MAD score = signed (logit - median_train) / (1.4826 MAD_train), training logits
              from the members' training raw scores
  params      384 (members) + 5 (combiner) = 389   (CADE-mini pairs: 398-406)
Stages (from pca8/quantum_hybrid_v5/; needs ../quantum_amp_8 and ../quantum_angle_8 with their test results):
  python3 run_quantum_hybrid_v5.py probe
  mkdir -p logs && sbatch run_validate.slurm        # 2 models x N_SPLITS x 3 folds validation trainings
  python3 run_quantum_hybrid_v5.py fit              # one combiner per split -> combiner/split<s>.json
  sbatch run_quantum_hybrid_v5.slurm                # applies the combiners to the saved test scores (fast)
  python3 run_quantum_hybrid_v5.py aggregate
"""
import os, sys, json, glob, argparse, importlib.util, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PCA_ROOT = os.path.dirname(HERE)
VAL_DIR, COMB_DIR, RES_DIR = os.path.join(HERE, 'validation'), os.path.join(HERE, 'combiner'), os.path.join(HERE, 'results')
FOLDS = 3
NAME = 'Quantum hybrid v5 (stacked angle-8 + amp-8, combiner learned on validation folds, 389 params)'
MEMBERS = ('amp', 'angle')


def _load(prefix):
    path = sorted(glob.glob(os.path.join(PCA_ROOT, f'{prefix}*', 'run_*.py')))
    if not path:
        raise SystemExit(f'Cannot find {prefix}* next to quantum_hybrid_v5 (expected in {PCA_ROOT})')
    spec = importlib.util.spec_from_file_location(prefix, path[0]); m = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, [path[0]]; spec.loader.exec_module(m); sys.argv = argv
    return m, os.path.dirname(path[0])


AMP, AMP_DIR = _load('quantum_amp_')
ANG, ANG_DIR = _load('quantum_angle_')
N_SPLITS, N_TASKS = AMP.N_SPLITS, AMP.N_TASKS


def zstd(raw, train_raw):
    med = np.median(train_raw); mad = 1.4826 * np.median(np.abs(train_raw - med))
    return (raw - med) / (mad if mad > 0 else 1e-12)


def features(z_ang, z_amp):
    return np.c_[z_ang, z_amp, z_ang * z_amp, np.abs(z_ang - z_amp)]


# ----------------------------------------------------------------------------- stage 1: validation trainings
def validate_job(job, data_dir, prefix, max_epochs):
    """job = model * (N_SPLITS * FOLDS) + split * FOLDS + fold;  model 0 = amp-8, 1 = angle-8.
    Training data of split s only; the real held-out family and the test set are never used."""
    per = N_SPLITS * FOLDS; model, rest = MEMBERS[job // per], job % per; s, f = rest // FOLDS, rest % FOLDS
    data = AMP.load_split(data_dir, prefix, s)
    Xf, yf, Xv, ov = AMP.validation_fold(data, s, f)
    print(f'VALIDATION job {job}: {model}-8 | split {s} ({data["family"]} untouched) | pseudo-drift fold {f}', flush=True)
    if model == 'amp':
        Ftr, (Fv,), info = AMP.train_embed(Xf, yf, [Xv], 42, max_epochs, tag='[val] ')
        raw, tr_raw = AMP.proto_detect(Ftr, yf, Fv)[0], AMP.proto_detect(Ftr, yf, Ftr)[0]
    else:
        v = ANG.VARIANTS[0]
        Ftr, (Fv,), info = ANG.train_embed(Xf, yf, [Xv], v, 42, max_epochs, tag='[val] ')
        raw, tr_raw = ANG.detect(Ftr, yf, Fv, v)[0], ANG.train_raw(Ftr, yf, v)
    os.makedirs(VAL_DIR, exist_ok=True)
    np.savez_compressed(os.path.join(VAL_DIR, f'{model}_s{s}_f{f}.npz'), raw=raw, train_raw=tr_raw, ood=ov)
    from sklearn.metrics import roc_auc_score
    print(f'  VALIDATION {model}-8 ROC_raw {roc_auc_score(ov, raw):.4f}  best epoch {info["best_epoch"]}', flush=True)


# ----------------------------------------------------------------------------- stage 2: one combiner per split
def fit():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    os.makedirs(COMB_DIR, exist_ok=True); ok = True
    print(f"  {'split':<7}{'folds':>6}{'val ROC: combiner':>19}{'mean-z':>9}{'angle':>8}{'amp':>8}   weights (z_ang, z_amp, prod, |diff|)")
    for s in range(N_SPLITS):
        Xs, ys, zm, za, zb = [], [], [], [], []
        for f in range(FOLDS):
            p = {m: os.path.join(VAL_DIR, f'{m}_s{s}_f{f}.npz') for m in MEMBERS}
            if not all(os.path.exists(x) for x in p.values()):
                continue
            A, G = np.load(p['amp']), np.load(p['angle'])
            assert np.array_equal(A['ood'], G['ood'])
            z_amp, z_ang = zstd(A['raw'], A['train_raw']), zstd(G['raw'], G['train_raw'])
            Xs.append(features(z_ang, z_amp)); ys.append(A['ood']); zm.append((z_ang + z_amp) / 2); za.append(z_ang); zb.append(z_amp)
        if len(Xs) < FOLDS:
            print(f'  split {s}: only {len(Xs)}/{FOLDS} folds complete -- not fitted'); ok = False; continue
        X, y = np.vstack(Xs), np.concatenate(ys)
        mu, sd = X.mean(0), X.std(0) + 1e-12
        lr = LogisticRegression(class_weight='balanced', max_iter=2000).fit((X - mu) / sd, y)
        logit = lr.decision_function((X - mu) / sd)
        comb = {'split': s, 'feature_mean': mu.tolist(), 'feature_std': sd.tolist(), 'coef': lr.coef_[0].tolist(),
                'intercept': float(lr.intercept_[0]), 'n_val_samples': int(len(y)), 'n_val_drift': int(y.sum()),
                'val_ROC_in_sample': float(roc_auc_score(y, logit))}
        json.dump(comb, open(os.path.join(COMB_DIR, f'split{s}.json'), 'w'), indent=2)
        r = lambda sc: roc_auc_score(y, np.concatenate(sc))
        print(f"  {s:<7}{len(Xs):>6}{comb['val_ROC_in_sample']:>19.3f}{r(zm):>9.3f}{r(za):>8.3f}{r(zb):>8.3f}   "
              + ', '.join(f'{c:+.2f}' for c in comb['coef']))
    print('\nvalidation ROC of the combiner is IN-SAMPLE (fitted on these folds): a sanity check, not a result.')
    if not ok:
        print('Some splits are not fitted -- finish the validation array first.')


def combiner_score(comb, z_ang, z_amp):
    X = (features(z_ang, z_amp) - np.array(comb['feature_mean'])) / np.array(comb['feature_std'])
    return X @ np.array(comb['coef']) + comb['intercept']


# ----------------------------------------------------------------------------- stage 3: test tasks
def run(task_id, data_dir, prefix):
    run_id, split, seed = AMP.task_info(task_id)
    cp = os.path.join(COMB_DIR, f'split{split}.json')
    if not os.path.exists(cp):
        raise SystemExit(f'No combiner for split {split} -- run "fit" after the validation array.')
    comb = json.load(open(cp)); data = AMP.load_split(data_dir, prefix, split)
    print(f'{NAME} | task {task_id} run {run_id} split {split} seed {seed} OOD={data["family"]}', flush=True)
    mem = {}
    for m, d in (('amp', AMP_DIR), ('angle', ANG_DIR)):
        t = os.path.join(d, 'results', f'run{run_id}', f'split{split}')
        c = np.genfromtxt(os.path.join(t, 'ood_scores.csv'), delimiter=',', names=True)
        mem[m] = {'raw': c['raw_score'], 'ood': c['is_ood'].astype(int), 'label': c['true_label'],
                  'train_raw': np.load(os.path.join(t, 'train_raw_scores.npy'))}
    assert np.array_equal(mem['amp']['label'], mem['angle']['label']) and np.array_equal(mem['amp']['ood'], data['is_ood'])
    za = {m: zstd(mem[m]['raw'], mem[m]['train_raw']) for m in MEMBERS}
    zt = {m: zstd(mem[m]['train_raw'], mem[m]['train_raw']) for m in MEMBERS}
    raw = combiner_score(comb, za['angle'], za['amp']); tr = combiner_score(comb, zt['angle'], zt['amp'])
    med = np.median(tr); mad = 1.4826 * np.median(np.abs(tr - med)); mad = mad if mad > 0 else 1e-12
    mad_te, mad_tr = (raw - med) / mad, (tr - med) / mad
    save_dir = os.path.join(RES_DIR, f'run{run_id}', f'split{split}'); os.makedirs(save_dir, exist_ok=True)
    meta = {'run_id': run_id, 'split_id': split, 'seed': seed,
            'input': 'saved scores of angle-8 and amp-8 (same seed and split), z-standardized by their training scores',
            'architecture': 'logistic-regression combiner on [z_angle, z_amp, z_angle*z_amp, |z_angle - z_amp|], '
                            f'fitted on split {split} validation folds only',
            'combiner': comb, 'n_trainable_params': 389, 'members_params': 384, 'combiner_params': 5}
    np.save(os.path.join(save_dir, 'train_raw_scores.npy'), tr); np.save(os.path.join(save_dir, 'train_anomaly_scores.npy'), mad_tr)
    s_ = AMP.save_task(save_dir, NAME, meta, raw, mad_te, data, [], None, None, None)
    e = s_['evaluation']
    print(f'  DONE  ROC_raw {e["threshold_free"]["ROC_AUC"]:.4f}  ROC_mad {e["ROC_AUC_mad"]:.4f}  -> {save_dir}', flush=True)


def probe():
    print(f'members: {AMP_DIR}  |  {ANG_DIR}')
    for m, d in (('amp-8', AMP_DIR), ('angle-8', ANG_DIR)):
        n = len(glob.glob(os.path.join(d, 'results', 'run*', 'split*', 'train_raw_scores.npy')))
        print(f'  {m}: {n}/{N_TASKS} test tasks with saved scores')
    print(f'  splits {N_SPLITS}, validation jobs {2 * N_SPLITS * FOLDS} (2 models x {N_SPLITS} splits x {FOLDS} folds), '
          f'test tasks {N_TASKS}; data {AMP.DEFAULT_DATA}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('cmd'); ap.add_argument('--job', type=int)
    ap.add_argument('--data-dir', default=AMP.DEFAULT_DATA); ap.add_argument('--prefix', default=AMP.DEFAULT_PREFIX)
    ap.add_argument('--max-epochs', type=int, default=AMP.MAX_EPOCHS); a = ap.parse_args()
    dd = os.path.expanduser(a.data_dir)
    if a.cmd == 'probe':
        probe()
    elif a.cmd == 'validate':
        validate_job(a.job, dd, a.prefix, a.max_epochs)
    elif a.cmd == 'fit':
        fit()
    elif a.cmd == 'aggregate':
        AMP.aggregate(RES_DIR, NAME)
    else:
        run(int(a.cmd), dd, a.prefix)
