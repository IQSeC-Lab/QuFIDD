#!/usr/bin/env python3
"""
plot_feature_maps.py -- Full_new_experiments / quantum   (standalone script)
===========================================================================
Feature-map plots for one task (default: task 0 = seed 42, first family), from files
the experiments already saved -- no retraining.

  1_circuit.png            the quantum feature map + trainable layers (qml.draw_mpl),
                           and 1_circuit.txt (text drawing)
  2_distance_matrices.png  pairwise distances between test samples, sorted by family
                           (OOD family last): PCA-5 input | CADE-mini full | CADE-mini
                           encoder-only | quantum UNTRAINED | quantum TRAINED
                           (quantum distance = sqrt(1 - fidelity); classical = Euclidean
                           in the latent space, divided by its max). Dark = close.
  3_bloch_vectors.png      each qubit's reduced state (<X>, <Y>, <Z>) for the test samples,
                           coloured by family -- untrained (top) vs trained (bottom)
  4_landscapes.png         OOD-score landscape over PCA components 1 and 2 (others fixed at
                           the training mean), for the same five representations; colour =
                           percentile of the score among training samples (1.0 = more unusual
                           than every training sample). Test points overlaid (OOD = black x).

Files used (all written by the run scripts):
  quantum/results/run{r}/split{s}/model_weights.pt, states.npz
  cade_mini_full/results/run{r}/split{s}/model_weights.pt
  cade_mini_encoder_only/results/run{r}/split{s}/model_weights.pt
PCA-5 is refit exactly as the run scripts do (full SVD on the training split, deterministic).

Usage (from Full_new_experiments/quantum/):
  python3 plot_feature_maps.py                 # task 0
  python3 plot_feature_maps.py 5               # any task 0-11
  python3 plot_feature_maps.py all             # all 12 tasks
Output: quantum/feature_maps/run{r}_split{s}/*.png
"""
import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SEEDS = {0: 42, 1: 123, 2: 456}
N_SPLITS, PCA_DIM = 4, 5
N_QUBITS, N_LAYERS, N_UPLOADS, INIT_STD = 5, 20, 4, 0.1
EVERY = N_LAYERS // N_UPLOADS
PER_CLASS = 60                     # test samples per family in the distance matrices / Bloch plots
GRID = 60                          # landscape grid resolution


# ----------------------------------------------------------------------------- data (same as run scripts)
def load_split(data_dir, prefix, s):
    d = np.load(os.path.join(os.path.expanduser(data_dir), f'{prefix}_{s}.npz'), allow_pickle=True)
    dense = lambda a: a.item().toarray().astype(np.float32) if a.shape == () else a.astype(np.float32)
    hf = d['heldout_family']; fam = str(hf.item()) if hf.shape == () else str(hf)
    Xtr, ytr, Xte, yte = dense(d['X_train']), d['y_train'].astype(np.int64), dense(d['X_test']), d['y_test'].astype(np.int64)
    known = sorted(int(c) for c in np.unique(ytr))
    ytr_c = np.array([known.index(int(c)) for c in ytr])
    return Xtr, ytr_c, Xte, yte, (~np.isin(yte, known)).astype(int), fam, known


# ----------------------------------------------------------------------------- models
def make_circuit():
    import pennylane as qml
    dev = qml.device('default.qubit', wires=N_QUBITS)

    @qml.qnode(dev, interface='torch')
    def circuit(angles, weights):
        for l in range(N_LAYERS):
            if l % EVERY == 0:
                for k in range(N_QUBITS):
                    qml.RY(angles[..., k], wires=k)
            for k in range(N_QUBITS):
                qml.RY(weights[l, k, 0], wires=k)
                qml.RZ(weights[l, k, 1], wires=k)
            for k in range(N_QUBITS - 1):
                qml.CNOT(wires=[k, k + 1])
            qml.CNOT(wires=[N_QUBITS - 1, 0])
            if l % 2 == 1:
                for k in range(0, N_QUBITS - 2, 2):
                    qml.CNOT(wires=[k, k + 2])
        return qml.state()
    return circuit


def q_states(circuit, weights, A, bs=512):
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(A), bs):
            psi = circuit(torch.from_numpy(A[i:i + bs].astype(np.float32)), weights)
            out.append((psi if psi.dim() == 2 else psi.unsqueeze(0)).numpy())
    return np.vstack(out)


def load_encoder(path, hidden, with_decoder, n_lat):
    """Rebuild the CADE-mini modules exactly as saved (ModuleList [encoder(, decoder)])."""
    import torch, torch.nn as nn
    dims = [PCA_DIM] + hidden + [n_lat]
    enc = []
    for i in range(len(dims) - 2):
        enc += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
    enc += [nn.Linear(dims[-2], dims[-1])]
    mods = [nn.Sequential(*enc)]
    if with_decoder:
        dec = []
        for i in range(len(dims) - 1, 1, -1):
            dec += [nn.Linear(dims[i], dims[i - 1]), nn.ReLU()]
        dec += [nn.Linear(dims[1], dims[0])]
        mods.append(nn.Sequential(*dec))
    ml = nn.ModuleList(mods)
    ml.load_state_dict(torch.load(path, map_location='cpu'))
    enc = ml[0]
    return lambda Z: enc(torch.from_numpy(np.asarray(Z, np.float32))).detach().numpy()


# ----------------------------------------------------------------------------- scores
def centroid_score(Ltr, ytr, L):
    return np.min([np.linalg.norm(L - Ltr[ytr == c].mean(0), axis=1) for c in np.unique(ytr)], axis=0)


def prototype_score(Str, ytr, S):
    return np.min([1 - (np.abs(S.conj() @ Str[ytr == c].T) ** 2).mean(1) for c in np.unique(ytr)], axis=0)


def bloch(states):
    """Reduced single-qubit Bloch vectors; wire 0 = most significant bit (PennyLane)."""
    n = N_QUBITS; psi = states.reshape(-1, *([2] * n)); out = np.zeros((len(states), n, 3))
    for k in range(n):
        m = np.moveaxis(psi, k + 1, 1).reshape(len(states), 2, -1)
        rho = np.einsum('bik,bjk->bij', m, m.conj())
        out[:, k, 0] = 2 * rho[:, 0, 1].real
        out[:, k, 1] = -2 * rho[:, 0, 1].imag
        out[:, k, 2] = (rho[:, 0, 0] - rho[:, 1, 1]).real
    return out


# ----------------------------------------------------------------------------- main per task
def run(task_id, data_dir, prefix):
    import torch
    r, s = task_id // N_SPLITS, task_id % N_SPLITS
    seed = SEEDS[r]
    Xtr, ytr, Xte, yte, is_ood, fam, known = load_split(data_dir, prefix, s)
    pca = PCA(n_components=PCA_DIM, svd_solver='full').fit(Xtr)
    Ztr, Zte = pca.transform(Xtr), pca.transform(Xte)
    out = os.path.join(HERE, 'feature_maps', f'run{r}_split{s}'); os.makedirs(out, exist_ok=True)
    qdir = os.path.join(HERE, 'results', f'run{r}', f'split{s}')
    ck = torch.load(os.path.join(qdir, 'model_weights.pt'), map_location='cpu', weights_only=False)
    mu, scale = np.asarray(ck['pca_mean']), float(ck['shared_scale'])
    angles = lambda Z: np.arctan((Z - mu) / scale)
    circuit = make_circuit()
    w_tr = ck['weights']
    torch.manual_seed(seed); w_un = torch.randn(N_LAYERS, N_QUBITS, 2) * INIT_STD   # exact initial weights
    print(f'task {task_id}: run {r} split {s} seed {seed} OOD={fam} -> {out}', flush=True)

    # ---- 1. circuit
    import pennylane as qml
    x0 = torch.from_numpy(angles(Ztr[:1])[0].astype(np.float32))
    with open(os.path.join(out, '1_circuit.txt'), 'w') as f:
        f.write(qml.draw(circuit, decimals=2)(x0, w_tr))
    fig, _ = qml.draw_mpl(circuit, decimals=None)(x0, w_tr)
    fig.suptitle(f'Quantum feature map + trainable layers ({N_QUBITS} qubits, {N_LAYERS} layers, '
                 f'{N_UPLOADS} data uploads RY(x_k)); 200 trainable angles', fontsize=14)
    fig.savefig(os.path.join(out, '1_circuit.png'), dpi=90, bbox_inches='tight'); plt.close(fig)

    # ---- representations of train / test
    Str_tr = np.load(os.path.join(qdir, 'states.npz'))['states_train']
    Ste_tr = q_states(circuit, w_tr, angles(Zte))
    Str_un, Ste_un = q_states(circuit, w_un, angles(Ztr)), q_states(circuit, w_un, angles(Zte))
    n_lat = len(np.unique(ytr))
    reps = {'PCA-5 input': ('euc', lambda Z: Z)}
    for name, folder, hid, dec in [('CADE-mini full', 'cade_mini_full', [11], True),
                                   ('CADE-mini encoder-only', 'cade_mini_encoder_only', [22], False)]:
        p = os.path.join(ROOT, folder, 'results', f'run{r}', f'split{s}', 'model_weights.pt')
        if os.path.exists(p):
            reps[name] = ('euc', load_encoder(p, hid, dec, n_lat))
        else:
            print(f'  (skipping {name}: {p} not found)')
    reps['Quantum untrained'] = ('q', (Str_un, Ste_un, w_un))
    reps['Quantum trained'] = ('q', (Str_tr, Ste_tr, w_tr))

    # sample test points per family, sorted: known families, then OOD
    rng = np.random.default_rng(0); order, ticks = [], []
    for lab in [c for c in known] + [int(np.unique(yte[is_ood == 1])[0])]:
        idx = np.where(yte == lab)[0]; idx = rng.choice(idx, min(PER_CLASS, len(idx)), replace=False)
        ticks.append((len(order), len(order) + len(idx), f'{"OOD " + fam if lab not in known else "class " + str(lab)}'))
        order += list(idx)
    order = np.array(order)

    # ---- 2. distance matrices
    fig, ax = plt.subplots(1, len(reps), figsize=(4.2 * len(reps), 4.4))
    for a, (name, (kind, obj)) in zip(ax, reps.items()):
        if kind == 'euc':
            L = obj(Zte[order]); D = np.linalg.norm(L[:, None] - L[None], axis=-1); D = D / D.max()
        else:
            S = obj[1][order]
            D = np.sqrt(np.clip(1 - np.abs(S.conj() @ S.T) ** 2, 0, 1))
        im = a.imshow(D, cmap='viridis', vmin=0, vmax=1)          # 0 (close) = dark
        for (b0, b1, lab) in ticks:
            a.axhline(b0 - 0.5, color='w', lw=0.6); a.axvline(b0 - 0.5, color='w', lw=0.6)
        a.set_xticks([(b0 + b1) / 2 for b0, b1, _ in ticks]); a.set_xticklabels([t for *_, t in ticks], fontsize=7, rotation=30)
        a.set_yticks([(b0 + b1) / 2 for b0, b1, _ in ticks]); a.set_yticklabels([t for *_, t in ticks], fontsize=7)
        a.set_title(name, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.015, label='distance (dark = close)')
    fig.suptitle(f'Pairwise distances between test samples (OOD = {fam}); good map: dark diagonal blocks, '
                 f'OOD block far from the known blocks', fontsize=11)
    fig.savefig(os.path.join(out, '2_distance_matrices.png'), dpi=130, bbox_inches='tight'); plt.close(fig)

    # ---- 3. Bloch vectors per qubit (untrained vs trained)
    fig = plt.figure(figsize=(4 * N_QUBITS, 8.2))
    cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for row, (lab, S) in enumerate([('untrained', Ste_un[order]), ('trained', Ste_tr[order])]):
        B = bloch(S); yl = yte[order]
        for k in range(N_QUBITS):
            a = fig.add_subplot(2, N_QUBITS, row * N_QUBITS + k + 1, projection='3d')
            u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:15j]
            a.plot_wireframe(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v), color='lightgrey', lw=0.3)
            for i, c in enumerate(known):
                m = yl == c; a.scatter(B[m, k, 0], B[m, k, 1], B[m, k, 2], s=6, color=cols[i], label=f'class {c}')
            m = ~np.isin(yl, known)
            a.scatter(B[m, k, 0], B[m, k, 1], B[m, k, 2], s=10, c='k', marker='x', label=f'OOD {fam}')
            a.set_title(f'qubit {k} ({lab})', fontsize=9); a.set_xlim(-1, 1); a.set_ylim(-1, 1); a.set_zlim(-1, 1)
            a.set_xlabel('<X>', fontsize=7); a.set_ylabel('<Y>', fontsize=7); a.set_zlabel('<Z>', fontsize=7)
            a.tick_params(labelsize=5)
            if row == 0 and k == 0:
                a.legend(fontsize=7, loc='upper left')
    fig.suptitle('Reduced single-qubit states of test samples (vectors inside the sphere = entangled qubits)', fontsize=11)
    fig.savefig(os.path.join(out, '3_bloch_vectors.png'), dpi=120, bbox_inches='tight'); plt.close(fig)

    # ---- 4. OOD-score landscapes over PC1 x PC2
    lo, hi = np.percentile(Ztr[:, :2], 1, axis=0), np.percentile(Ztr[:, :2], 99, axis=0)
    pad = 0.15 * (hi - lo); lo, hi = lo - pad, hi + pad
    g1, g2 = np.meshgrid(np.linspace(lo[0], hi[0], GRID), np.linspace(lo[1], hi[1], GRID))
    Zg = np.tile(Ztr.mean(0), (GRID * GRID, 1)); Zg[:, 0], Zg[:, 1] = g1.ravel(), g2.ravel()
    fig, ax = plt.subplots(1, len(reps), figsize=(4.2 * len(reps), 4.2))
    for a, (name, (kind, obj)) in zip(ax, reps.items()):
        if kind == 'euc':
            Ltr = obj(Ztr); sg, st = centroid_score(Ltr, ytr, obj(Zg)), centroid_score(Ltr, ytr, Ltr)
        else:
            Strain, _, w = obj
            sg, st = prototype_score(Strain, ytr, q_states(circuit, w, angles(Zg))), prototype_score(Strain, ytr, Strain)
        pct = np.searchsorted(np.sort(st), sg) / len(st)
        im = a.imshow(pct.reshape(GRID, GRID), origin='lower', extent=[lo[0], hi[0], lo[1], hi[1]],
                      aspect='auto', cmap='magma', vmin=0, vmax=1)
        known_m = is_ood == 0
        a.scatter(Zte[known_m, 0], Zte[known_m, 1], s=2, c='c', alpha=0.4, label='test known')
        a.scatter(Zte[~known_m, 0], Zte[~known_m, 1], s=4, c='w', marker='x', alpha=0.6, label=f'test OOD')
        a.set_xlim(lo[0], hi[0]); a.set_ylim(lo[1], hi[1]); a.set_xlabel('PC1')
        a.set_title(name, fontsize=10)
    ax[0].legend(fontsize=7, loc='upper right'); ax[0].set_ylabel('PC2')
    fig.colorbar(im, ax=ax, fraction=0.015, label='OOD score percentile vs training samples')
    fig.suptitle(f'OOD-score landscape over PC1 x PC2 (PC3-5 fixed at the training mean); OOD = {fam}', fontsize=11)
    fig.savefig(os.path.join(out, '4_landscapes.png'), dpi=130, bbox_inches='tight'); plt.close(fig)
    print(f'  saved 1_circuit.png/.txt, 2_distance_matrices.png, 3_bloch_vectors.png, 4_landscapes.png', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('task_id', nargs='?', default='0', help="0-11 or 'all' (default 0)")
    p.add_argument('--data-dir', default='~/projects/OOD/Data/cade_lamda_4class_vt')
    p.add_argument('--prefix', default='drebin_new')
    a = p.parse_args()
    for t in (range(3 * N_SPLITS) if a.task_id == 'all' else [int(a.task_id)]):
        run(t, a.data_dir, a.prefix)
