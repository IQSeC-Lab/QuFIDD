#!/usr/bin/env python3
"""
pennylane_hybrid_v7_p702.py -- STANDALONE PennyLane PCA-8 quantum hybrid v7, scaled to ~700 params (cycling, a=0.5)
======================================================================================================================
Self-contained: does not import our simulator or any of our model scripts. The circuit is built and simulated with
PennyLane (default.qubit, torch interface, backprop, parameter broadcasting); data, PCA, angle map, CADE pairing,
losses, early stopping, prototypes and the raw-distance evaluation are re-implemented here with the SAME settings and
the SAME random seeds as capacity_matched_8fam/pca8/quantum_hybrid_v7_cycle_a05. Same design as that model (cycling
re-uploads, a = 0.5, same losses and training), scaled from 384 to 702 parameters: more layers AND more re-uploads.

Model
  input       PCA-8 (fit on the training split only)
  encoding    L2-normalized PCA-8 amplitude-encoded on 3 qubits (qml.StatePrep)
  circuit     62 layers of RX, RY, RZ on every qubit + CNOT ring (0->1, 1->2, 2->0)
  re-uploads  24 angle re-uploads RY(a*phi + b) before layers 2,5,7,10,12,15,17,20,22,25,28,30,
              33,35,38,40,43,45,48,50,53,56,58,61, cycling through all 8 PCs in groups of 3 (each PC exactly 9 times);
              phi = arctan((PC - mean) / std(PC1)); a starts at 0.5, b at 0
  params      62 x 3 x 3 + 24 x 3 x 2 = 558 + 144 = 702   (v7: 384)
  training    pair loss on D = 1 - fidelity (CADE pairing, margin 1) + prototype cross-entropy on fidelities (tau 0.1);
              pure-state prototypes (top eigenvector of each family's density matrix) recomputed every epoch;
              Adam lr 1e-3, batch 64, early stopping (patience 20, relative 1e-4, max 250), best weights restored
  score       raw distance = 1 - max_c |<v_c|psi>|^2 (fidelity distance to the nearest prototype)

  python3 pennylane_hybrid_v7_p702.py <task 0-(3*N_SPLITS-1)>   one task (N_SPLITS held-out families x 3 seeds)
  python3 pennylane_hybrid_v7_p702.py aggregate           mean +- std over the finished tasks
Outputs: <results-dir>/run<r>/split<s>/{summary.json, ood_scores.csv, states.npz, model_weights.pt, training_loss.csv,
         plots/{training_loss, roc_pr_raw, raw_score_hist, latent_kpca}.png}; aggregate -> aggregate_summary.txt,
         aggregate.csv, aggregate_plots/{ROC_AUC_raw,PR_AUC_raw,TF_F1}_per_family.png
"""
import os, sys, csv, json, time, copy, argparse
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score

# ----------------------------------------------------------------------------- settings (identical to our run)
SEEDS = {0: 42, 1: 123, 2: 456}
N_SPLITS = 8
PCA_DIM = 8
BATCH_SIZE, SIMILAR_RATIO = 64, 0.25
MAX_EPOCHS, PATIENCE, REL_MIN_DELTA, LR = 250, 20, 1e-4, 0.001
INIT_STD, MARGIN, TAU, LAMBDA_P = 0.1, 1.0, 0.1, 0.0   # ABLATION: pair (CADE-style) loss only
N_Q, L_Q, Q_INIT = 3, 62, 0.5
UP_Q = (2, 5, 7, 10, 12, 15, 17, 20, 22, 25, 28, 30, 33, 35, 38, 40, 43, 45, 48, 50, 53, 56, 58, 61)
GROUPS = [tuple([p % 8 for p in range(72)][3 * u:3 * u + 3]) for u in range(24)]    # (0,1,2),(3,4,5),(6,7,0),...
assert len(UP_Q) == len(GROUPS) == 24 and max(UP_Q) < L_Q
NAME = 'Quantum hybrid v7 cycle a=0.5 (PCA-8, 3 qubits, 62 layers, 24 re-uploads, 702 params) [PennyLane] [ABLATION pair loss only]'
DEFAULT_DATA = '/home/snowm001/projects/OOD/Data2/EMBER/cade_ember2018_cadesplit_1000_k8'


def set_seed(seed):
    import torch
    np.random.seed(seed); torch.manual_seed(seed)


def task_info(t):
    run = t // N_SPLITS
    return run, t % N_SPLITS, SEEDS[run]


# ----------------------------------------------------------------------------- data
def load_split(data_dir, prefix, s):
    d = np.load(os.path.join(os.path.expanduser(data_dir), f'{prefix}_{s}.npz'), allow_pickle=True)
    dense = lambda a: a.item().toarray().astype(np.float32) if a.shape == () else a.astype(np.float32)
    hf = d['heldout_family']; fam = str(hf.item()) if hf.shape == () else str(hf)
    Xtr, ytr, Xte, yte = dense(d['X_train']), d['y_train'].astype(np.int64), dense(d['X_test']), d['y_test'].astype(np.int64)
    known = sorted(int(c) for c in np.unique(ytr)); remap = {c: i for i, c in enumerate(known)}
    return {'Xtr': Xtr, 'ytr': np.array([remap[int(c)] for c in ytr], dtype=np.int64), 'Xte': Xte, 'yte': yte,
            'is_ood': (~np.isin(yte, known)).astype(int), 'family': fam}


class HybInput:
    """PCA-8 (training data only) -> [8 amplitudes (L2-normalized) | 8 angles arctan((PC - mean) / std(PC1))]."""
    def fit(self, Xtr):
        self.pca = PCA(n_components=PCA_DIM, svd_solver='full').fit(Xtr)
        Z = self.pca.transform(Xtr); self.mu, self.scale = Z.mean(0), float(Z[:, 0].std()) + 1e-8
        return self

    def __call__(self, X):
        Z = self.pca.transform(X)
        amp = (Z / np.linalg.norm(Z, axis=1, keepdims=True).clip(min=1e-8)).astype(np.float32)
        return np.hstack([amp, np.arctan((Z - self.mu) / self.scale).astype(np.float32)])


def epoch_batches(X, y):
    """CADE's pairing (github.com/whyisyoung/CADE): first half random, second half 25% same-family / rest other."""
    half = BATCH_SIZE // 2; bc = int(X.shape[0] / half); num_sim = int(BATCH_SIZE * SIMILAR_RATIO)
    bx = np.zeros([bc, BATCH_SIZE, X.shape[1]], dtype=np.float32); by = np.zeros([bc, BATCH_SIZE], dtype=np.int64)
    idx = np.random.permutation(X.shape[0])
    bx[:, :half] = np.split(X[idx[: bc * half]], bc); by[:, :half] = np.split(y[idx[: bc * half]], bc)
    cls = [np.where(y == c)[0] for c in range(len(np.unique(y)))]; ncls = [np.where(y != c)[0] for c in range(len(np.unique(y)))]
    for b in range(bc):
        for m in range(num_sim):
            j = np.random.choice(cls[by[b, m]], 1)[0]; bx[b, m + half], by[b, m + half] = X[j], y[j]
        for m in range(num_sim, half):
            j = np.random.choice(ncls[by[b, m]], 1)[0]; bx[b, m + half], by[b, m + half] = X[j], y[j]
    return bc, bx, by


class EarlyStopper:
    def __init__(self):
        self.best, self.best_epoch, self.since, self.state = np.inf, 0, 0, None

    def step(self, epoch, loss, state_fn):
        if loss < self.best * (1.0 - REL_MIN_DELTA):
            self.best, self.best_epoch, self.since = loss, epoch, 0; self.state = copy.deepcopy(state_fn())
        else:
            self.since += 1
        return self.since >= PATIENCE


# ----------------------------------------------------------------------------- PennyLane circuit
def make_circuit():
    import pennylane as qml
    dev = qml.device('default.qubit', wires=N_Q)

    @qml.qnode(dev, interface='torch', diff_method='backprop')
    def circuit(amp, ang, W, a, b):
        qml.StatePrep(amp, wires=range(N_Q), normalize=True)            # amplitude encoding (batched)
        u = 0
        for l in range(L_Q):
            if l in UP_Q:                                                # angle re-upload (per-sample angles)
                for k in range(N_Q):
                    qml.RY(a[u, k] * ang[:, GROUPS[u][k]] + b[u, k], wires=k)
                u += 1
            for k in range(N_Q):                                         # trainable layer
                qml.RX(W[l, k, 0], wires=k); qml.RY(W[l, k, 1], wires=k); qml.RZ(W[l, k, 2], wires=k)
            for k in range(N_Q - 1):
                qml.CNOT(wires=[k, k + 1])
            qml.CNOT(wires=[N_Q - 1, 0])
        return qml.state()
    return circuit


def pure_prototypes(S, y, C):
    """Top eigenvector of each family's density matrix rho_c = mean |psi><psi| (a pure, phase-invariant prototype)."""
    import torch
    St = torch.as_tensor(S); yt = torch.as_tensor(y, device=St.device); V = []
    for c in range(C):
        P = St[yt == c]; rho = P.T @ P.conj() / len(P)
        V.append(torch.linalg.eigh(rho)[1][:, -1].conj())
    return torch.stack(V)


# ----------------------------------------------------------------------------- plots (raw distance only)
def plot_task(out, hist, best_epoch, raw, ood, Str, ytr, Ste, family):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, precision_recall_curve
    from sklearn.decomposition import KernelPCA
    pd = os.path.join(out, 'plots'); os.makedirs(pd, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 3.5))                               # training loss
    ax.plot(range(1, len(hist) + 1), hist); ax.axvline(best_epoch, color='k', ls='--', lw=1, label=f'best epoch {best_epoch}')
    ax.set_xlabel('epoch'); ax.set_ylabel('training loss'); ax.set_title(f'{family} held out'); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(pd, 'training_loss.png'), dpi=130); plt.close(fig)
    fpr, tpr, _ = roc_curve(ood, raw); pr, rc, _ = precision_recall_curve(ood, raw)
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.8))                        # ROC + PR of the raw distance
    axs[0].plot(fpr, tpr); axs[0].plot([0, 1], [0, 1], 'k:', lw=1)
    axs[0].set_xlabel('FPR'); axs[0].set_ylabel('TPR'); axs[0].set_title(f'ROC (raw), AUC {roc_auc_score(ood, raw):.3f}')
    axs[1].plot(rc, pr); axs[1].axhline(ood.mean(), color='k', ls=':', lw=1)
    axs[1].set_xlabel('recall'); axs[1].set_ylabel('precision'); axs[1].set_title(f'PR (raw), AP {average_precision_score(ood, raw):.3f}')
    fig.suptitle(f'{family} held out', fontsize=10); fig.tight_layout(); fig.savefig(os.path.join(pd, 'roc_pr_raw.png'), dpi=130); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 3.5))                               # raw-score distributions
    bins = np.linspace(raw.min(), raw.max(), 50)
    ax.hist(raw[ood == 0], bins=bins, alpha=0.6, density=True, label='known families (test)')
    ax.hist(raw[ood == 1], bins=bins, alpha=0.6, density=True, label=f'{family} (drifting)')
    ax.set_xlabel('raw distance 1 - max_c F(v_c, psi)'); ax.set_ylabel('density'); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(pd, 'raw_score_hist.png'), dpi=130); plt.close(fig)
    r = np.random.default_rng(0); itr = np.sort(r.choice(len(Str), min(800, len(Str)), replace=False))
    ite = np.sort(np.random.default_rng(1).choice(len(Ste), min(1500, len(Ste)), replace=False))
    K = lambda A, B: np.abs(A.conj() @ B.T) ** 2                           # fidelity kernel
    kp = KernelPCA(n_components=2, kernel='precomputed').fit(K(Str[itr], Str[itr]))
    Ztr, Zte = kp.transform(K(Str[itr], Str[itr])), kp.transform(K(Ste[ite], Str[itr]))
    fig, ax = plt.subplots(figsize=(5, 4.2))                               # latent space
    ax.scatter(Ztr[:, 0], Ztr[:, 1], c=ytr[itr], cmap='tab10', s=4, alpha=0.5, linewidths=0)
    o = ood[ite].astype(bool); ax.scatter(Zte[o, 0], Zte[o, 1], c='k', s=5, alpha=0.6, linewidths=0, label=f'{family} (drifting, test)')
    ax.set_xticks([]); ax.set_yticks([]); ax.legend(fontsize=8, loc='best')
    ax.set_title('kernel PCA of the fidelity kernel\ncoloured: known families (train)', fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(pd, 'latent_kpca.png'), dpi=130); plt.close(fig)


# ----------------------------------------------------------------------------- training + evaluation
def best_f1(raw, is_ood):
    order = np.argsort(-raw, kind='stable'); tp = np.cumsum(is_ood[order]); k = np.arange(1, len(raw) + 1)
    p, r = tp / k, tp / is_ood.sum(); f = np.where(p + r > 0, 2 * p * r / (p + r + 1e-12), 0.0); b = int(np.argmax(f))
    return {'F1': float(f[b]), 'Precision': float(p[b]), 'Recall': float(r[b]),
            'FPR': float((k[b] - tp[b]) / (len(raw) - is_ood.sum())), 'FNR': float(1 - r[b]),
            'Effort': float(k[b] / is_ood.sum()), 'k': int(k[b])}


def run_task(task, data_dir, prefix, results_dir, max_epochs):
    import torch, torch.optim as optim, pennylane as qml
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '4')))
    run_id, split, seed = task_info(task); data = load_split(data_dir, prefix, split)
    print(f'{NAME} | task {task} run {run_id} split {split} seed {seed} OOD={data["family"]} '
          f'| pennylane {qml.__version__}, torch {torch.__version__}', flush=True)
    circuit = make_circuit()
    inp = HybInput().fit(data['Xtr']); Atr = inp(data['Xtr']); ytr = data['ytr']; C = len(np.unique(ytr))
    set_seed(seed)                                                       # same initial weights as our run
    P = {'W': torch.nn.Parameter(torch.randn(L_Q, N_Q, 3) * INIT_STD),
         'a': torch.nn.Parameter(torch.full((len(UP_Q), N_Q), Q_INIT)), 'b': torch.nn.Parameter(torch.zeros(len(UP_Q), N_Q))}
    opt = optim.Adam(list(P.values()), lr=LR); n_params = int(sum(p.numel() for p in P.values()))
    st = lambda A: circuit(A[:, :PCA_DIM].double(), A[:, PCA_DIM:].double(), P['W'], P['a'], P['b'])
    states = lambda A: torch.cat([st(torch.from_numpy(A[i:i + 512])) for i in range(0, len(A), 512)])
    print(f'  {n_params} trainable params', flush=True)
    stop, hist, t0 = EarlyStopper(), [], time.time()
    snap = lambda: {k: v.detach().clone() for k, v in P.items()}; stop.state = snap()
    for epoch in range(1, max_epochs + 1):
        with torch.no_grad():
            V = pure_prototypes(states(Atr), ytr, C)                     # prototypes refreshed every epoch
        bc, bx, by = epoch_batches(Atr, ytr); ls = []
        for b in range(bc):
            A, yb = torch.from_numpy(bx[b]), torch.from_numpy(by[b])
            opt.zero_grad()
            S = st(A); h = S.shape[0] // 2
            ov = (S[:h].conj() * S[h:]).sum(-1); d = 1.0 - (ov.real ** 2 + ov.imag ** 2)
            same = (yb[:h] == yb[h:]).double()
            pair = (same * d + (1 - same) * torch.relu(MARGIN - d)).mean()
            proto = torch.nn.functional.cross_entropy((S @ V.T).abs() ** 2 / TAU, yb)
            loss = pair + LAMBDA_P * proto
            loss.backward(); opt.step(); ls.append(loss.item())
        hist.append(float(np.mean(ls))); done = stop.step(epoch, hist[-1], snap)
        print(f'  epoch {epoch:3d} loss {hist[-1]:.5f} best {stop.best:.5f} (ep {stop.best_epoch}) '
              f'no_improve {stop.since} ({time.time() - t0:.0f}s)', flush=True)
        if done:
            break
    with torch.no_grad():
        for k in P:
            P[k].copy_(stop.state[k])
        Str = states(Atr).numpy(); Ste = states(inp(data['Xte'])).numpy()
    V = pure_prototypes(Str, ytr, C).numpy()
    raw = (1.0 - np.abs(Ste @ V.T) ** 2).min(1); ood = data['is_ood']
    tf = best_f1(raw, ood); tf['ROC_AUC'] = float(roc_auc_score(ood, raw)); tf['PR_AUC'] = float(average_precision_score(ood, raw))
    out = os.path.join(results_dir, f'run{run_id}', f'split{split}'); os.makedirs(out, exist_ok=True)
    summary = {'model': NAME, 'simulator': 'pennylane default.qubit', 'run_id': run_id, 'split_id': split, 'seed': seed,
               'ood_family': data['family'], 'n_trainable_params': n_params, 'epochs_run': len(hist),
               'best_epoch': stop.best_epoch, 'best_loss': stop.best, 'train_time_sec': round(time.time() - t0, 1),
               'evaluation': {'threshold_free': tf}}
    json.dump(summary, open(os.path.join(out, 'summary.json'), 'w'), indent=2, default=float)
    with open(os.path.join(out, 'ood_scores.csv'), 'w') as f:
        f.write('raw_score,is_ood,true_label\n')
        for s_, o, y in zip(raw, ood, data['yte']):
            f.write(f'{s_},{o},{y}\n')
    with open(os.path.join(out, 'training_loss.csv'), 'w') as f:
        f.write('epoch,loss\n' + ''.join(f'{i + 1},{v}\n' for i, v in enumerate(hist)))
    np.savez_compressed(os.path.join(out, 'states.npz'), states_train_r0=Str, states_test_r0=Ste,
                        y_train=ytr, y_test=data['yte'])
    torch.save({k: v.detach() for k, v in P.items()}, os.path.join(out, 'model_weights.pt'))
    plot_task(out, hist, stop.best_epoch, raw, ood, Str, ytr, Ste, data['family'])
    print(f'  DONE  ROC_raw {tf["ROC_AUC"]:.4f}  PR_raw {tf["PR_AUC"]:.4f}  TF_F1 {tf["F1"]:.4f}  -> {out}', flush=True)


def aggregate(results_dir):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    rows = []
    for t in range(3 * N_SPLITS):
        r, s, _ = task_info(t); f = os.path.join(results_dir, f'run{r}', f'split{s}', 'summary.json')
        if os.path.exists(f):
            j = json.load(open(f)); tf = j['evaluation']['threshold_free']
            rows.append({'run': r, 'split': s, 'family': j['ood_family'], 'n_params': j['n_trainable_params'],
                         'best_epoch': j['best_epoch'], 'epochs_run': j['epochs_run'], 'train_time_sec': j['train_time_sec'],
                         'ROC_AUC_raw': tf['ROC_AUC'], 'PR_AUC_raw': tf['PR_AUC'], 'TF_F1': tf['F1'],
                         'Precision': tf['Precision'], 'Recall': tf['Recall'], 'FPR': tf['FPR'],
                         'FNR': tf.get('FNR', 1 - tf['Recall']), 'Effort': tf.get('Effort', float('nan'))})
    if not rows:
        print('no finished tasks in', results_dir); return
    mets = ['TF_F1', 'ROC_AUC_raw', 'PR_AUC_raw', 'Precision', 'Recall', 'FPR', 'FNR', 'Effort']
    fams = list(dict.fromkeys(r['family'] for r in rows))
    L = ['=' * 100, f'{NAME} -- AGGREGATED ({os.path.abspath(results_dir)})', '=' * 100,
         f'Completed: {len(rows)}/{3 * N_SPLITS}   Missing: ' + (str([t for t in range(3 * N_SPLITS) if not any(x['run'] == t // N_SPLITS and x['split'] == t % N_SPLITS for x in rows)]) if len(rows) < 3 * N_SPLITS else 'none'),
         f'  {"family":12s} {"n":>3} ' + ' '.join(f'{m:>15s}' for m in mets)]
    for fam in fams + ['ALL']:
        sel = [x for x in rows if fam == 'ALL' or x['family'] == fam]
        L.append(f'  {fam:12s} {len(sel):3d} ' + ' '.join(f'{np.mean([x[m] for x in sel]):7.3f}+-{np.std([x[m] for x in sel]):.3f}' for m in mets))
    L.append(f'  Params: {rows[0]["n_params"]}   best epochs: {[x["best_epoch"] for x in rows]}   (raw distance only; std = population std)')
    open(os.path.join(results_dir, 'aggregate_summary.txt'), 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
    with open(os.path.join(results_dir, 'aggregate.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    pd = os.path.join(results_dir, 'aggregate_plots'); os.makedirs(pd, exist_ok=True)
    for m in ('ROC_AUC_raw', 'PR_AUC_raw', 'TF_F1'):
        fig, ax = plt.subplots(figsize=(7, 3.6)); x = np.arange(len(fams))
        mu = [np.mean([r[m] for r in rows if r['family'] == f]) for f in fams]; sd = [np.std([r[m] for r in rows if r['family'] == f]) for f in fams]
        ax.bar(x, mu, yerr=sd, capsize=3, color='0.6')
        for i, f in enumerate(fams):
            ax.scatter([i] * len([r for r in rows if r['family'] == f]), [r[m] for r in rows if r['family'] == f], color='k', s=10, zorder=3)
        ax.axhline(np.mean([r[m] for r in rows]), color='k', ls='--', lw=1, label=f'mean {np.mean([r[m] for r in rows]):.3f}')
        if m == 'ROC_AUC_raw':
            ax.axhline(0.5, color='k', ls=':', lw=1)
        ax.set_xticks(x); ax.set_xticklabels(fams, rotation=30, ha='right', fontsize=8); ax.set_ylabel(m); ax.legend(fontsize=8)
        ax.set_title(f'{m} per held-out family (bars: mean +- std over seeds; dots: seeds)', fontsize=9)
        fig.tight_layout(); fig.savefig(os.path.join(pd, f'{m}_per_family.png'), dpi=130); plt.close(fig)
    print(f'Saved: {results_dir}/aggregate_summary.txt, aggregate.csv, aggregate_plots/*.png')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('task'); ap.add_argument('--data-dir', default=DEFAULT_DATA); ap.add_argument('--prefix', default='drebin_new')
    ap.add_argument('--results-dir', default='results'); ap.add_argument('--max-epochs', type=int, default=MAX_EPOCHS)
    a = ap.parse_args()
    if a.task == 'aggregate':
        aggregate(a.results_dir)
    else:
        run_task(int(a.task), a.data_dir, a.prefix, a.results_dir, a.max_epochs)


if __name__ == '__main__':
    main()
