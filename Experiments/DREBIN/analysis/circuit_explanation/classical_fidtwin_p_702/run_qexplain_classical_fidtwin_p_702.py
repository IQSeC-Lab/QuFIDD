#!/usr/bin/env python3
"""
run_qexplain_classical_fidtwin_p_702.py -- quantum-native drift explanation for
the fidelity twin, plain (M3 only); Drebin
===============================================================================================================
Standalone. Re-loads the trained model of one task (seed run --run), recomputes every state with an exact state-vector
simulator of the same circuit (checked against PennyLane to 1e-15 and against the saved drift scores), and explains the
drift scores of ALL test samples with three methods:

M1  Re-upload decomposition (quantum models only). The trainable layers apply the same unitary to every sample and
    fidelity is invariant under a shared unitary, so the fidelity distance of a sample to the (pure) prototype of a
    family, recomputed from the training states at the same depth, can change ONLY at the 24 data re-uploads. With
    c* = the family nearest to the sample at the output, the drift score splits exactly into
        s(x) = s_0(x) + sum_u Delta_u(x),
    s_0 = distance after the amplitude encoding alone (cosine geometry of the PCA input), Delta_u = change across
    re-upload u. The script verifies numerically that the distance is constant between re-uploads and that the sum
    is exact. Each Delta_u is split among its 3 single-qubit RY gates (one PC each) by an exact 3-player Shapley
    value; a gate left out is replaced by the identity, which is equivalent to any sample-independent angle (a shared
    rotation leaves fidelities unchanged), so the split has no baseline choice. s_0 is split among the 8 PCs by an
    exact 8-player Shapley value (256 coalitions) with the reference sample x_ref (training sample of c* closest to
    its prototype) as baseline: v(empty) = s_0(x_ref). Hence
        s(x) = s_0(x_ref) + sum_j [ phi_amp_j(x) + phi_reup_j(x) ]        (exact, per PC j)
    and, since PCA is linear, each PC's attribution is passed on to the input features in proportion to their share of
    the PC's deviation from x_ref:  feat_i = sum_j phi_j * W_ji (x_i - xref_i) / (z_j - zref_j).
M2  Pauli decomposition (quantum models only). For 3-qubit states F(psi, v) = (1/8) sum_P <P>_psi <P>_v over the 64
    Pauli strings, so the deviation of the drift score from the mean distance of the family's training members is
        s(x) - mean_train d_c* = sum_{P != I} (1/8) <P>_v ( <P>_rho_c* - <P>_psi ),
    an exact sum over 63 terms, grouped by weight (1-, 2-, 3-qubit terms) and by qubit support. These are Pauli
    expectation values, i.e. measurable on hardware (e.g. classical shadows) without state tomography. Also reported:
    deviation of each qubit's Bloch vector from the family mean, and single-qubit purities (entanglement).
M3  Prototype-mixture profile (quantum models and fidelity twins): fidelities to all C prototypes, the share of the state
    outside the span of the prototypes (novelty residual), and the entropy of the normalised fidelity profile; for the
    drifting family, the mean profile says which known families it resembles.

Evaluations
  E1  CADE metrics (Sec. 5) for the M1 feature ranking on the same inspected samples as drift_analysis_cade (top-k
      features with the k of the CADE explanation when its results exist, else k = 10): distance of the perturbed sample
      to c*, crossing of the MAD boundary (median + 3.5 MAD) and of the FPR-10% boundary; random baseline (CADE style).
  E2  Insertion curves: known-family percentile of the drift score after replacing the top-k features (k grid) of each
      drifting sample by x_ref; M1 ranking vs. gradient x difference vs. random; area under the curve (lower = better).
  E3  Re-upload / PC ablation: ROC-AUC change when one re-upload (or all gates of one PC) is removed, and its Spearman
      correlation with the drift-vs-known gap of Delta_u (or of phi_j).

Usage (from this folder):  python3 run_qexplain_classical_fidtwin_p_702.py <split> | all | aggregate
Outputs: results/split<s>/{summary.json, decomposition.csv, pc_attribution.npz, pauli.npz, mixture.npz, eval.json,
         plots/*.png}, results/aggregate_summary.txt
"""
import os, sys, csv, json, time, argparse, itertools, math
import numpy as np

MODEL_KIND = 'twin_p'          # quantum | quantum_fixed | twin_c | twin_p | twin_r
MODEL_TAG = 'classical_fidtwin_p_702'
DEFAULT_MODEL_RESULTS = '../../twin_pca8_p702/classical_fidtwin_p_702/results'
DEFAULT_CADE_RESULTS = '../../drift_analysis_cade/classical_fidtwin_p_702/results'      # drift_analysis_cade results of the same model (optional)
DEFAULT_DATA = '/home/snowm001/projects/OOD/Data2/Drebin/cade_drebin'
DATASET = 'Drebin'
BINARY = DATASET == 'Drebin'
N_SPLITS, PCA_DIM, N_Q, L_Q = 8, 8, 3, 62
QUANTUM = MODEL_KIND.startswith('quantum')
UP_Q = (2, 5, 7, 10, 12, 15, 17, 20, 22, 25, 28, 30, 33, 35, 38, 40, 43, 45, 48, 50, 53, 56, 58, 61)
GROUPS = ([(0, 1, 2)] * 24 if MODEL_KIND == 'quantum_fixed'
          else [tuple([p % 8 for p in range(72)][3 * u:3 * u + 3]) for u in range(24)])
MAD_T, MAD_B = 3.5, 1.4826
K_GRID = (0, 1, 2, 5, 10, 20, 50, 100, 200)
DREBIN_NAMES = ['FakeInstaller', 'DroidKungFu', 'Plankton', 'GinMaster', 'BaseBridge', 'Iconosys', 'Kmin', 'FakeDoc']
EMBER_GROUPS = [('byte_histogram', 256), ('byte_entropy_histogram', 256), ('strings', 104), ('general', 10),
                ('header', 62), ('section', 255), ('imports', 1280), ('exports', 128), ('data_directories', 30)]


# ----------------------------------------------------------------------------- data
def load_split(data_dir, prefix, s):
    d = np.load(os.path.join(os.path.expanduser(data_dir), f'{prefix}_{s}.npz'), allow_pickle=True)
    dense = lambda a: a.item().toarray().astype(np.float32) if a.shape == () else a.astype(np.float32)
    Xtr, ytr, Xte, yte = dense(d['X_train']), d['y_train'].astype(np.int64), dense(d['X_test']), d['y_test'].astype(np.int64)
    known = sorted(int(c) for c in np.unique(ytr)); remap = {c: i for i, c in enumerate(known)}
    fams = [str(x) for x in d['families']] if 'families' in d.files else (
        DREBIN_NAMES if max(int(yte.max()), int(ytr.max())) < len(DREBIN_NAMES) else None)
    if 'heldout_family' in d.files:
        hf = d['heldout_family']; fam = str(hf.item()) if hf.shape == () else str(hf)
    else:
        miss = sorted(set(np.unique(yte).tolist()) - set(known))
        fam = fams[int(miss[0])] if (fams and len(miss) == 1) else 'label_' + '_'.join(map(str, miss))
    name_of = lambda c: fams[c] if fams else str(c)
    return {'Xtr': Xtr, 'ytr': np.array([remap[int(c)] for c in ytr], dtype=np.int64), 'Xte': Xte, 'yte': yte,
            'is_ood': (~np.isin(yte, known)).astype(int), 'family': fam,
            'known_names': [name_of(c) for c in known], 'feature_index': d['feature_index'] if 'feature_index' in d.files else None}


def feature_names(n_feat, split, feature_index, names_dir):
    if DATASET == 'Drebin' and names_dir:
        f = os.path.join(names_dir, f'drebin_new{split}_train_selected_features.txt')
        if os.path.exists(f):
            names = [l.strip() for l in open(f)]
            if len(names) == n_feat:
                return names
    if DATASET == 'EMBER' and feature_index is not None:
        bounds = np.cumsum([n for _, n in EMBER_GROUPS]); starts = bounds - np.array([n for _, n in EMBER_GROUPS])
        out = []
        for c in feature_index:
            g = int(np.searchsorted(bounds, c, side='right')); out.append(f'{EMBER_GROUPS[g][0]}[{int(c - starts[g])}]')
        return out
    return [f'f{j}' for j in range(n_feat)]


class TorchInput:
    """PCA-8 fitted on the training split (as in training) as a torch map x -> z -> (amplitudes, angles)."""
    def __init__(self, Xtr, dtype=np.float32):
        from sklearn.decomposition import PCA
        import torch
        self.pca = PCA(n_components=PCA_DIM, svd_solver='full').fit(Xtr.astype(dtype))
        Z = self.pca.transform(Xtr); mu, scale = Z.mean(0), float(Z[:, 0].std()) + 1e-8
        self.W0 = torch.tensor(self.pca.components_, dtype=torch.float64)
        self.m = torch.tensor(self.pca.mean_, dtype=torch.float64)
        self.mu0 = torch.tensor(mu, dtype=torch.float64); self.scale = scale
        self.set_signs(np.ones(PCA_DIM))

    def set_signs(self, signs):
        import torch
        self.signs = np.asarray(signs, dtype=float); sg = torch.tensor(self.signs, dtype=torch.float64)
        self.W = self.W0 * sg[:, None]; self.mu = self.mu0 * sg

    def z(self, X):
        return (X - self.m) @ self.W.T

    def amp(self, Z):
        import torch
        return Z / torch.linalg.vector_norm(Z, dim=1, keepdim=True).clamp(min=1e-8)

    def ang(self, Z):
        import torch
        return torch.arctan((Z - self.mu) / self.scale)


# ----------------------------------------------------------------------------- exact state-vector simulator
def _rx(t):
    import torch
    c, s = torch.cos(t / 2), torch.sin(t / 2)
    return torch.stack([torch.stack([c + 0j, -1j * s]), torch.stack([-1j * s, c + 0j])])


def _ry(t):
    import torch
    c, s = torch.cos(t / 2), torch.sin(t / 2)
    return torch.stack([torch.stack([c, -s]), torch.stack([s, c])]).to(torch.complex128)


def _rz(t):
    import torch
    return torch.diag(torch.stack([torch.exp(-0.5j * t), torch.exp(0.5j * t)]))


def _ry_batch(theta):
    import torch
    c, s = torch.cos(theta / 2), torch.sin(theta / 2)
    return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2).to(torch.complex128)


def _apply1(psi, G, k):
    import torch
    p = psi.movedim(k + 1, 1)
    p = torch.einsum('xy,bykl->bxkl', G, p) if G.dim() == 2 else torch.einsum('bxy,bykl->bxkl', G, p)
    return p.movedim(1, k + 1)


def _cnot(psi, c, t):
    p = psi.clone(); idx = [slice(None)] * 4; idx[c + 1] = 1
    p[tuple(idx)] = psi[tuple(idx)].flip((t if t < c else t - 1) + 1)
    return p


def layer(psi, W, l):
    for k in range(N_Q):
        psi = _apply1(psi, _rx(W[l, k, 0]), k); psi = _apply1(psi, _ry(W[l, k, 1]), k); psi = _apply1(psi, _rz(W[l, k, 2]), k)
    return _cnot(_cnot(_cnot(psi, 0, 1), 1, 2), 2, 0)


def reupload(psi, ang, a, b, u, gates=(0, 1, 2)):
    for k in gates:
        psi = _apply1(psi, _ry_batch(a[u, k] * ang[:, GROUPS[u][k]] + b[u, k]), k)
    return psi


def trace(amp, ang, W, a, b):
    """Run the circuit; returns amp state, states just before and just after every re-upload, and the output state."""
    import torch
    psi = (amp / amp.norm(dim=1, keepdim=True)).to(torch.complex128).reshape(-1, 2, 2, 2)
    s0 = psi.reshape(-1, 8).clone(); before, after = [], []; u = 0
    for l in range(L_Q):
        if l in UP_Q:
            before.append(psi.reshape(-1, 8).clone()); psi = reupload(psi, ang, a, b, u); after.append(psi.reshape(-1, 8).clone()); u += 1
        psi = layer(psi, W, l)
    return {'amp': s0, 'before': before, 'after': after, 'out': psi.reshape(-1, 8)}


def final_states(amp, ang, W, a, b):
    import torch
    psi = (amp / amp.norm(dim=1, keepdim=True)).to(torch.complex128).reshape(-1, 2, 2, 2); u = 0
    for l in range(L_Q):
        if l in UP_Q:
            psi = reupload(psi, ang, a, b, u); u += 1
        psi = layer(psi, W, l)
    return psi.reshape(-1, 8)


def build_embed(model_dir, inp):
    import torch
    P = torch.load(os.path.join(model_dir, 'model_weights.pt'), map_location='cpu', weights_only=False)
    if QUANTUM:
        W, a, b = (P[k].double() for k in ('W', 'a', 'b'))
        def embed(X):
            Z = inp.z(X); return final_states(inp.amp(Z), inp.ang(Z), W, a, b)
        return embed, (W, a, b)
    W1, b1, W2, b2 = (P[k].double() for k in ('W1', 'b1', 'W2', 'b2'))
    def embed(X):
        Z = inp.z(X); amp = inp.amp(Z); A = torch.cat([amp, inp.ang(Z)], 1)
        g = torch.tanh(A @ W1.T + b1) @ W2.T + b2
        if MODEL_KIND == 'twin_c':
            zc = torch.complex(amp + g[:, :PCA_DIM], g[:, PCA_DIM:])
        elif MODEL_KIND == 'twin_p':
            zc = torch.complex(g[:, :PCA_DIM], g[:, PCA_DIM:])
        else:
            zc = torch.complex(amp + g, torch.zeros_like(g))
        return zc / torch.linalg.vector_norm(zc, dim=1, keepdim=True).clamp(min=1e-8)
    return embed, None


def prototypes(E, y, C):
    import torch
    out = []
    for c in range(C):
        S = E[torch.as_tensor(y == c)]; rho = S.T @ S.conj() / len(S)
        out.append(torch.linalg.eigh(rho)[1][:, -1].conj())
    return torch.stack(out)


def dist_all(E, P):
    return 1.0 - (E @ P.T).abs() ** 2


def dist_c(E, P, c):
    # P holds conj(prototype) (as in the training scripts), so <v_c|psi> = sum_i P[c]_i psi_i
    return 1.0 - (E * P[c]).sum(-1).abs() ** 2


def reload_model(data, model_dir):
    import torch
    Xtr = torch.tensor(data['Xtr'], dtype=torch.float64); Xte = torch.tensor(data['Xte'], dtype=torch.float64)
    ytr = data['ytr']; C = len(np.unique(ytr))
    saved = np.loadtxt(os.path.join(model_dir, 'ood_scores.csv'), delimiter=',', skiprows=1, usecols=0)

    def check(inp):
        embed, w = build_embed(model_dir, inp)
        with torch.no_grad():
            Etr, Ete = embed(Xtr), embed(Xte); P = prototypes(Etr, ytr, C)
        rel = float(np.max(np.abs(dist_all(Ete, P).numpy().min(1) - saved)) / (np.max(np.abs(saved)) + 1e-12))
        return rel, embed, w
    tried = []
    for dt in (np.float32, np.float64):
        inp = TorchInput(data['Xtr'], dt); rel, embed, w = check(inp); tried.append(f'{np.dtype(dt).name} {rel:.1e}')
        if rel <= 1e-3:
            break
    if rel > 1e-3:
        rs = np.random.default_rng(123); te_s = np.sort(rs.choice(len(Xte), min(150, len(Xte)), replace=False))
        tr_s = np.concatenate([rs.choice(np.flatnonzero(ytr == c), min(150, int((ytr == c).sum())), replace=False) for c in range(C)])
        rank = lambda v: np.argsort(np.argsort(v)); cand = []
        emb_s, _ = build_embed(model_dir, inp)
        for sg in itertools.product((1.0, -1.0), repeat=PCA_DIM):
            inp.set_signs(sg)
            with torch.no_grad():
                E = emb_s(torch.cat([Xtr[tr_s], Xte[te_s]])); P = prototypes(E[:len(tr_s)], ytr[tr_s], C)
                sc = dist_all(E[len(tr_s):], P).numpy().min(1)
            cand.append((np.corrcoef(rank(sc), rank(saved[te_s]))[0, 1], sg))
        for rho, sg in sorted(cand, key=lambda t: -t[0])[:6]:
            inp.set_signs(sg); rel, embed, w = check(inp); tried.append(f'signs {"".join("+" if v > 0 else "-" for v in sg)} {rel:.1e}')
            if rel <= 1e-3:
                break
    if rel > 1e-3:
        raise SystemExit(f'reloaded model does not reproduce the saved scores ({"; ".join(tried)})')
    return inp, embed, w, Xtr, Xte, rel, tried


def roc_auc(score, is_ood):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(is_ood, score))


def pauli_matrices():
    import torch
    I = np.eye(2); X = np.array([[0, 1], [1, 0]]); Y = np.array([[0, -1j], [1j, 0]]); Z = np.diag([1, -1])
    mats, labels = [], []
    for s in itertools.product('IXYZ', repeat=N_Q):
        M = np.array([[1.0]])
        for ch in s:
            M = np.kron(M, {'I': I, 'X': X, 'Y': Y, 'Z': Z}[ch])         # wire 0 = most significant (PennyLane)
        mats.append(M); labels.append(''.join(s))
    return torch.tensor(np.array(mats), dtype=torch.complex128), labels


def expect(E, Pm):
    """<P>_psi for all states E (N,8) and Pauli matrices Pm (64,8,8) -> (N,64) real."""
    import torch
    return torch.einsum('ni,pij,nj->np', E.conj(), Pm, E).real


# ----------------------------------------------------------------------------- one split
def explain_split(split, data_dir, prefix, model_results, cade_results, run_id, n_samples, out_root, names_dir, seed=0):
    import torch
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '4'))); torch.manual_seed(seed)
    t_start = time.time()
    data = load_split(data_dir, prefix, split); ytr, is_ood = data['ytr'], data['is_ood']
    model_dir = os.path.join(model_results, f'run{run_id}', f'split{split}')
    inp, embed, w, Xtr, Xte, rel, tried = reload_model(data, model_dir)
    C = len(np.unique(ytr)); out = os.path.join(out_root, f'split{split}'); os.makedirs(os.path.join(out, 'plots'), exist_ok=True)
    with torch.no_grad():
        Etr, Ete = embed(Xtr), embed(Xte); P = prototypes(Etr, ytr, C)
        Dtr, Dte = dist_all(Etr, P).numpy(), dist_all(Ete, P).numpy()
    score = Dte.min(1); cstar = Dte.argmin(1); cs = torch.as_tensor(cstar)
    auc = roc_auc(score, is_ood)
    print(f'{MODEL_TAG} split {split} (withheld {data["family"]}): reload rel. diff {rel:.1e}; ROC-AUC {auc:.3f}', flush=True)
    summ = {'model': MODEL_TAG, 'dataset': DATASET, 'split': split, 'withheld_family': data['family'], 'run': run_id,
            'reload_rel_diff': rel, 'roc_auc': auc, 'n_test': int(len(score)), 'n_drift': int(is_ood.sum())}
    gap = lambda v: float(np.mean(v[is_ood == 1]) - np.mean(v[is_ood == 0]))      # drift-vs-known gap of a quantity
    ref_idx = np.array([np.flatnonzero(ytr == c)[Dtr[ytr == c, c].argmin()] for c in range(C)])
    med = np.array([np.median(Dtr[ytr == c, c]) for c in range(C)])
    mad = np.array([max(MAD_B * np.median(np.abs(Dtr[ytr == c, c] - med[c])), 1e-12) for c in range(C)])
    lowerbound = med + MAD_T * mad; thr10 = float(np.quantile(score[is_ood == 0], 0.9))
    known_sorted = np.sort(score[is_ood == 0]); pct = lambda s: np.searchsorted(known_sorted, s) / len(known_sorted)

    # ---------------- M3: prototype-mixture profile (all fidelity models)
    with torch.no_grad():
        Fall = (Ete @ P.T).abs().pow(2).numpy()
        Q, _ = torch.linalg.qr(P.conj().T)                                     # orthonormal basis of the prototype span
        resid = (1.0 - (Ete @ Q.conj()).abs().pow(2).sum(1)).clamp(min=0).numpy()
    prof = Fall / np.maximum(Fall.sum(1, keepdims=True), 1e-12)
    ent = -(prof * np.log(np.maximum(prof, 1e-12))).sum(1) / np.log(C)
    resemble = prof[is_ood == 1].mean(0)
    np.savez_compressed(os.path.join(out, 'mixture.npz'), fidelities=Fall.astype(np.float32), residual=resid.astype(np.float32),
                        entropy=ent.astype(np.float32), is_ood=is_ood, nearest=cstar, known_names=np.array(data['known_names']))
    summ['M3'] = {'auc_residual': roc_auc(resid, is_ood), 'auc_entropy': roc_auc(ent, is_ood),
                  'auc_score': auc, 'residual_gap': gap(resid), 'entropy_gap': gap(ent),
                  'drifting_family_resembles': {data['known_names'][c]: float(resemble[c]) for c in np.argsort(-resemble)},
                  'nearest_family_of_drifting': {data['known_names'][c]: int((cstar[is_ood == 1] == c).sum()) for c in range(C)}}

    if not QUANTUM:
        json.dump(summ, open(os.path.join(out, 'summary.json'), 'w'), indent=2)
        print(f'  M3: AUC residual {summ["M3"]["auc_residual"]:.3f}, entropy {summ["M3"]["auc_entropy"]:.3f}', flush=True)
        return

    W, a, b = w
    with torch.no_grad():
        Ztr, Zte = inp.z(Xtr), inp.z(Xte)
        Atr, Ate, Gtr, Gte = inp.amp(Ztr), inp.amp(Zte), inp.ang(Ztr), inp.ang(Zte)
        Ttr, Tte = trace(Atr, Gtr, W, a, b), trace(Ate, Gte, W, a, b)
        # ---------------- M1: re-upload decomposition
        P0 = prototypes(Ttr['amp'], ytr, C); s0 = dist_c(Tte['amp'], P0, cs).numpy()
        d_b, d_a, Pb_all = [], [], []
        for u in range(24):
            Pb = prototypes(Ttr['before'][u], ytr, C); Pa = prototypes(Ttr['after'][u], ytr, C)
            d_b.append(dist_c(Tte['before'][u], Pb, cs).numpy()); d_a.append(dist_c(Tte['after'][u], Pa, cs).numpy()); Pb_all.append(Pb)
        d_b, d_a = np.stack(d_b, 1), np.stack(d_a, 1); Delta = d_a - d_b
        seg_err = max(float(np.abs(d_b[:, 0] - s0).max()), float(np.abs(d_b[:, 1:] - d_a[:, :-1]).max()),
                      float(np.abs(d_a[:, -1] - score).max()))
        sum_err = float(np.abs(s0 + Delta.sum(1) - score).max())
        # Shapley split of every Delta_u among its 3 gates (identity baseline = any shared angle)
        phi_gate = np.zeros((len(score), 24, 3))
        subsets = [S for r in range(4) for S in itertools.combinations(range(3), r)]
        for u in range(24):
            v = {}
            for S in subsets:
                if len(S) == 0:
                    v[S] = d_b[:, u]; continue
                tb = reupload(Ttr['before'][u].reshape(-1, 2, 2, 2), Gtr, a, b, u, S).reshape(-1, 8)
                te = reupload(Tte['before'][u].reshape(-1, 2, 2, 2), Gte, a, b, u, S).reshape(-1, 8)
                v[S] = dist_c(te, prototypes(tb, ytr, C), cs).numpy()
            for k in range(3):
                for S in subsets:
                    if k in S:
                        continue
                    wgt = math.factorial(len(S)) * math.factorial(2 - len(S)) / 6.0
                    phi_gate[:, u, k] += wgt * (v[tuple(sorted(S + (k,)))] - v[S])
        phi_reup = np.zeros((len(score), PCA_DIM))
        for u in range(24):
            for k in range(3):
                phi_reup[:, GROUPS[u][k]] += phi_gate[:, u, k]
        # Shapley split of s_0 among the 8 PCs, baseline = x_ref of c* (amplitude encoding only)
        Zref = Ztr[torch.as_tensor(ref_idx[cstar])]
        vals = {}
        for mask in range(256):
            m = torch.tensor([(mask >> j) & 1 for j in range(PCA_DIM)], dtype=torch.float64)
            Zm = Zte * m + Zref * (1 - m)
            vals[mask] = dist_c(inp.amp(Zm).to(torch.complex128), P0, cs).numpy()
        phi_amp = np.zeros((len(score), PCA_DIM)); n = PCA_DIM
        for j in range(n):
            for mask in range(256):
                if (mask >> j) & 1:
                    continue
                sz = bin(mask).count('1'); wgt = math.factorial(sz) * math.factorial(n - sz - 1) / math.factorial(n)
                phi_amp[:, j] += wgt * (vals[mask | (1 << j)] - vals[mask])
        base = vals[0]; amp_err = float(np.abs(base + phi_amp.sum(1) - s0).max())
        phi_pc = phi_amp + phi_reup
    summ['M1'] = {'segment_invariance_max_err': seg_err, 'decomposition_max_err': sum_err, 'amp_shapley_max_err': amp_err,
                  'auc_s0': roc_auc(s0, is_ood), 'auc_score': auc,
                  'gap_score': gap(score), 'gap_s0': gap(s0), 'gap_reuploads': gap(Delta.sum(1)),
                  'gap_per_reupload': [gap(Delta[:, u]) for u in range(24)],
                  'gap_per_pc_amp': [gap(phi_amp[:, j]) for j in range(PCA_DIM)],
                  'gap_per_pc_reup': [gap(phi_reup[:, j]) for j in range(PCA_DIM)],
                  'mean_Delta_drift': Delta[is_ood == 1].mean(0).tolist(), 'mean_Delta_known': Delta[is_ood == 0].mean(0).tolist(),
                  's0_mean_drift': float(s0[is_ood == 1].mean()), 's0_mean_known': float(s0[is_ood == 0].mean())}
    print(f'  M1: invariance err {seg_err:.1e}, decomposition err {sum_err:.1e}; AUC s0 {summ["M1"]["auc_s0"]:.3f} -> score {auc:.3f}; '
          f'gap score {summ["M1"]["gap_score"]:.3f} = s0 {summ["M1"]["gap_s0"]:.3f} + re-uploads {summ["M1"]["gap_reuploads"]:.3f}', flush=True)
    with open(os.path.join(out, 'decomposition.csv'), 'w', newline='') as fh:
        wr = csv.writer(fh); wr.writerow(['test_index', 'is_drift', 'nearest_family', 'score', 's0'] + [f'Delta_{u + 1}' for u in range(24)])
        for i in range(len(score)):
            wr.writerow([i, int(is_ood[i]), data['known_names'][cstar[i]], f'{score[i]:.6g}', f'{s0[i]:.6g}'] + [f'{x:.5g}' for x in Delta[i]])
    np.savez_compressed(os.path.join(out, 'pc_attribution.npz'), phi_amp=phi_amp.astype(np.float32), phi_reup=phi_reup.astype(np.float32),
                        phi_gate=phi_gate.astype(np.float32), base=base.astype(np.float32), is_ood=is_ood)

    # ---------------- M2: Pauli decomposition at the output
    Pm, labels = pauli_matrices()
    with torch.no_grad():
        ex_te = expect(Ete, Pm).numpy(); ex_v = expect(P.conj(), Pm).numpy(); ex_tr = expect(Etr, Pm).numpy()
    ex_rho = np.stack([ex_tr[ytr == c].mean(0) for c in range(C)])
    delta = (1.0 / 8.0) * ex_v[cstar] * (ex_rho[cstar] - ex_te)                 # (N,64); identity term is 0
    mean_member = np.array([Dtr[ytr == c, c].mean() for c in range(C)])
    pauli_err = float(np.abs(delta.sum(1) - (score - mean_member[cstar])).max())
    weight = np.array([sum(ch != 'I' for ch in l) for l in labels])
    support = [tuple(q for q, ch in enumerate(l) if ch != 'I') for l in labels]
    by_w = np.stack([delta[:, weight == k].sum(1) for k in (1, 2, 3)], 1)
    supports = [s for r in (1, 2, 3) for s in itertools.combinations(range(N_Q), r)]
    by_s = np.stack([delta[:, [i for i, s in enumerate(support) if s == S]].sum(1) for S in supports], 1)
    bloch = np.stack([ex_te[:, [labels.index(''.join(ch if q == k else 'I' for q in range(N_Q))) for ch in 'XYZ']] for k in range(N_Q)], 1)
    bloch_tr = np.stack([ex_tr[:, [labels.index(''.join(ch if q == k else 'I' for q in range(N_Q))) for ch in 'XYZ']] for k in range(N_Q)], 1)
    bloch_mean = np.stack([bloch_tr[ytr == c].mean(0) for c in range(C)])
    bloch_dev = np.linalg.norm(bloch - bloch_mean[cstar], axis=2)               # (N, 3 qubits)
    purity = 0.5 * (1 + (bloch ** 2).sum(2))                                    # single-qubit purity (1 = unentangled)
    np.savez_compressed(os.path.join(out, 'pauli.npz'), delta=delta.astype(np.float32), labels=np.array(labels), by_weight=by_w,
                        by_support=by_s, supports=np.array([str(s) for s in supports]), bloch_dev=bloch_dev, purity=purity, is_ood=is_ood)
    summ['M2'] = {'pauli_decomposition_max_err': pauli_err,
                  'gap_by_weight': {f'{k}-qubit': gap(by_w[:, k - 1]) for k in (1, 2, 3)},
                  'gap_by_support': {str(S): gap(by_s[:, i]) for i, S in enumerate(supports)},
                  'gap_bloch_dev': [gap(bloch_dev[:, k]) for k in range(N_Q)],
                  'mean_purity_drift': purity[is_ood == 1].mean(0).tolist(), 'mean_purity_known': purity[is_ood == 0].mean(0).tolist(),
                  'top_pauli_terms_by_gap': sorted(((labels[p], gap(delta[:, p])) for p in range(64) if weight[p] > 0), key=lambda t: -abs(t[1]))[:8]}
    print(f'  M2: Pauli err {pauli_err:.1e}; gap by weight ' + ', '.join(f'{k}: {v:.3f}' for k, v in summ['M2']['gap_by_weight'].items()), flush=True)

    # ---------------- E3: ablation of re-uploads / PCs and agreement with M1
    def auc_with(a_mod):
        with torch.no_grad():
            Et, Ee = final_states(Atr, Gtr, W, a_mod, b), final_states(Ate, Gte, W, a_mod, b)
            Pp = prototypes(Et, ytr, C); return roc_auc(dist_all(Ee, Pp).numpy().min(1), is_ood)
    def no_reup(u):                                  # identity re-upload (scale 0 + shared offset = shared rotation)
        am = a.clone(); am[u] = 0.0; return am
    dauc_u = np.array([auc - auc_with(no_reup(u)) for u in range(24)])
    dauc_pc = []
    for j in range(PCA_DIM):
        am = a.clone()
        for u in range(24):
            for k in range(3):
                if GROUPS[u][k] == j:
                    am[u, k] = 0.0
        dauc_pc.append(auc - auc_with(am))
    from scipy.stats import spearmanr
    gu = np.array(summ['M1']['gap_per_reupload']); gp = np.array(summ['M1']['gap_per_pc_reup'])
    summ['E3'] = {'dauc_per_reupload': dauc_u.tolist(), 'dauc_per_pc': dauc_pc,
                  'spearman_gapDelta_vs_dauc': float(spearmanr(gu, dauc_u).correlation),
                  'spearman_gapPhiReup_vs_daucPC': float(spearmanr(gp, dauc_pc).correlation) if np.std(dauc_pc) > 0 else None}
    print(f'  E3: Spearman(gap Delta_u, dAUC_u) = {summ["E3"]["spearman_gapDelta_vs_dauc"]:.2f}', flush=True)

    # ---------------- feature attribution for the explained (inspected) samples, E1 and E2
    order = np.argsort(-score, kind='stable'); o = is_ood[order]
    tp = np.cumsum(o); kk = np.arange(1, len(o) + 1); prec, rec = tp / kk, tp / max(o.sum(), 1)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec + 1e-12), 0); best_cnt = int(f1.argmax()) + 1
    inspected = order[:best_cnt]
    rng = np.random.default_rng(seed)                                        # same subset as drift_analysis_cade
    idx = inspected if not (n_samples and best_cnt > n_samples) else np.sort(rng.choice(inspected, n_samples, replace=False))
    kmap, cade_rows = {}, {}
    cf = os.path.join(cade_results, f'split{split}', 'explanations.csv') if cade_results else ''
    if cf and os.path.exists(cf):
        for r in csv.DictReader(open(cf)):
            kmap[int(r['test_index'])] = int(r['k']); cade_rows[int(r['test_index'])] = r
    ks = np.array([kmap.get(int(i), 10) for i in idx])
    x = Xte[idx]; xr = Xtr[torch.as_tensor(ref_idx[cstar[idx]])]; dx = (x - xr).numpy()
    Wpca = inp.W.numpy(); dz = (inp.z(x) - inp.z(xr)).numpy()
    share = np.einsum('ji,ni->nji', Wpca, dx) / np.where(np.abs(dz) > 1e-9, dz, np.inf)[:, :, None]
    feat = np.einsum('nj,nji->ni', phi_pc[idx], share)                          # per-feature attribution (N_expl, F)
    ci = torch.as_tensor(cstar[idx])
    xg = x.clone().requires_grad_(True); dist_c(embed(xg), P, ci).sum().backward()
    grad = (-xg.grad * (xr - x)).numpy()                                       # first-order distance decrease
    xn, xrn = x.numpy(), xr.numpy()

    def perturb(i, feats):
        v = xn[i].copy(); feats = np.asarray(feats, int)
        if len(feats):
            v[feats] = np.where(xn[i][feats] == 0, 1.0, 0.0) if BINARY else xrn[i][feats]
        return v

    def rank(scores_i, i):                                                     # candidates: features that differ from x_ref
        cand = np.flatnonzero(dx[i] != 0); return cand[np.argsort(-scores_i[cand], kind='mergesort')]

    def evaluate(sel):
        Xp = torch.tensor(np.stack([perturb(i, sel[i]) for i in range(len(idx))]))
        with torch.no_grad():
            E = embed(Xp); dp = dist_c(E, P, ci).numpy(); smin = dist_all(E, P).numpy().min(1)
        return dp, dp <= lowerbound[cstar[idx]], smin <= thr10, smin
    d0 = score[idx]
    sel_m1 = [rank(feat[i], i)[:ks[i]] for i in range(len(idx))]
    r_m1 = evaluate(sel_m1)
    e1 = {'n': int(len(idx)), 'k_mean': float(ks.mean()), 'k_from_cade': bool(kmap), 'd_orig': float(d0.mean()),
          'M1': {'d': float(r_m1[0].mean()), 'rel_d': float(r_m1[0].mean() / max(d0.mean(), 1e-12)),
                 'success_mad': float(r_m1[1].mean()), 'success_fpr10': float(r_m1[2].mean())}}
    dd, s1, s2 = [], [], []
    for _ in range(20):
        r = evaluate([rng.choice(xn.shape[1], ks[i], replace=False) for i in range(len(idx))]); dd.append(r[0]); s1.append(r[1]); s2.append(r[2])
    e1['random'] = {'d': float(np.mean(dd)), 'rel_d': float(np.mean(dd) / max(d0.mean(), 1e-12)),
                    'success_mad': float(np.mean(s1)), 'success_fpr10': float(np.mean(s2))}
    if cade_rows:
        cr = [cade_rows[int(i)] for i in idx if int(i) in cade_rows and cade_rows[int(i)]['d_cade']]
        if cr:
            e1['cade_mask_same_samples'] = {'d': float(np.mean([float(r['d_cade']) for r in cr])), 'n': len(cr)}
    summ['E1'] = e1
    # E2 insertion curves on the explained drifting samples
    dr = [i for i in range(len(idx)) if is_ood[idx[i]] == 1]
    curves = {}
    for name, sc in (('M1', feat), ('gradient', grad), ('random', None)):
        cur = []
        for kq in K_GRID:
            sel = [(rank(sc[i], i) if sc is not None else rng.permutation(np.flatnonzero(dx[i] != 0)))[:kq] if i in dr else [] for i in range(len(idx))]
            smin = evaluate(sel)[3]; cur.append(float(pct(smin[dr]).mean()) if dr else float('nan'))
        curves[name] = cur
    summ['E2'] = {'k_grid': list(K_GRID), 'curves_known_percentile': curves,
                  'auc': {k: float(np.mean(v)) for k, v in curves.items()}, 'n_drifting': len(dr)}
    print(f'  E1: rel d M1 {e1["M1"]["rel_d"]:.3f} / random {e1["random"]["rel_d"]:.3f}; success(MAD) {e1["M1"]["success_mad"]:.1%}; '
          f'E2 insertion AUC ' + ', '.join(f'{k} {v:.3f}' for k, v in summ['E2']['auc'].items()), flush=True)
    names = feature_names(xn.shape[1], split, data['feature_index'], names_dir)
    with open(os.path.join(out, 'top_features.txt'), 'w') as fh:              # case study, first 5 drifting samples
        for i in dr[:5]:
            fh.write(f'# test sample {int(idx[i])} ({data["family"]}), nearest family {data["known_names"][cstar[idx[i]]]}, score {score[idx[i]]:.4f} '
                     f'= s0(x_ref) {base[idx[i]]:.4f} + PCs {np.round(phi_pc[idx[i]], 4).tolist()}\n')
            for f_ in rank(feat[i], i)[:15]:
                fh.write(f'{names[f_]},attribution {feat[i, f_]:+.4g},value {xn[i, f_]:g},reference {xrn[i, f_]:g}\n')
            fh.write('\n')
    summ['time_s'] = time.time() - t_start
    json.dump(summ, open(os.path.join(out, 'summary.json'), 'w'), indent=2)
    plot_split(out, summ, Delta, s0, phi_amp, phi_reup, by_w, is_ood)


def plot_split(out, summ, Delta, s0, phi_amp, phi_reup, by_w, is_ood):
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    except Exception:
        return
    fam = summ['withheld_family']
    fig, axs = plt.subplots(1, 2, figsize=(11, 3), gridspec_kw={'width_ratios': [1.1, 1]})
    xs = np.arange(25); comp = np.c_[s0, Delta]
    axs[0].bar(xs, comp[is_ood == 1].mean(0) - comp[is_ood == 0].mean(0), color=['tab:gray'] + ['tab:blue'] * 24)
    axs[0].set_xticks(xs); axs[0].set_xticklabels(['s0'] + [str(u + 1) for u in range(24)], fontsize=6); axs[0].axhline(0, color='k', lw=0.5)
    axs[0].set_xlabel('amplitude encoding (s0) and re-upload u'); axs[0].set_ylabel('drift - known gap of contribution')
    traj = np.cumsum(comp, 1)                                     # score after the amplitude encoding and after each re-upload
    for sel, lab, col in ((is_ood == 1, f'drifting ({fam})', 'tab:red'), (is_ood == 0, 'known', 'tab:green')):
        mu, sd = traj[sel].mean(0), traj[sel].std(0)
        axs[1].plot(xs, mu, color=col, label=lab); axs[1].fill_between(xs, mu - sd, mu + sd, color=col, alpha=0.15)
    axs[1].set_xticks(xs[::2]); axs[1].set_xticklabels((['s0'] + [str(u + 1) for u in range(24)])[::2], fontsize=6)
    axs[1].set_xlabel('position in the circuit'); axs[1].set_ylabel('drift score so far'); axs[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'plots', 'reupload_decomposition.png'), dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 3)); xs = np.arange(PCA_DIM)
    for off, ph, lab in ((-0.3, phi_amp, 'amplitude'), (0.0, phi_reup, 're-uploads')):
        ax.bar(xs + off, ph[is_ood == 1].mean(0) - ph[is_ood == 0].mean(0), 0.3, label=lab)
    ax.set_xticks(xs); ax.set_xticklabels([f'PC{j + 1}' for j in range(PCA_DIM)]); ax.axhline(0, color='k', lw=0.5)
    ax.set_ylabel('drift - known gap'); ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(os.path.join(out, 'plots', 'pc_attribution_gap.png'), dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(['1-qubit', '2-qubit', '3-qubit'], [by_w[is_ood == 1, k].mean() - by_w[is_ood == 0, k].mean() for k in range(3)])
    ax.set_ylabel('drift - known gap'); ax.set_title('Pauli terms by weight', fontsize=9); fig.tight_layout()
    fig.savefig(os.path.join(out, 'plots', 'pauli_weight_gap.png'), dpi=150); plt.close(fig)
    if 'E2' in summ:
        fig, ax = plt.subplots(figsize=(4.5, 3))
        for k, v in summ['E2']['curves_known_percentile'].items():
            ax.plot(summ['E2']['k_grid'], v, marker='o', label=k)
        ax.set_xscale('symlog'); ax.set_xlabel('features replaced by x_ref'); ax.set_ylabel('known-family percentile of score')
        ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(out, 'plots', 'insertion_curves.png'), dpi=150); plt.close(fig)


def aggregate(out_root):
    rows = [json.load(open(os.path.join(out_root, f'split{s}', 'summary.json'))) for s in range(N_SPLITS)
            if os.path.exists(os.path.join(out_root, f'split{s}', 'summary.json'))]
    if not rows:
        print('no explained splits in', out_root); return
    m = lambda f: float(np.mean([f(r) for r in rows]))
    L = [f'{MODEL_TAG} ({DATASET}) -- quantum-native drift explanation, {len(rows)}/{N_SPLITS} splits', '',
         f'  {"withheld":14s} {"AUC":>6} {"AUC res":>8} {"AUC ent":>8}' + (f' {"AUC s0":>7} {"gap":>6} {"gap s0":>7} {"gap reup":>9} {"err":>8} {"rho E3":>7} {"relD M1":>8} {"relD rnd":>9} {"ins M1":>7} {"ins grad":>8} {"ins rnd":>8}' if QUANTUM else '')]
    for r in rows:
        line = f'  {r["withheld_family"][:14]:14s} {r["roc_auc"]:6.3f} {r["M3"]["auc_residual"]:8.3f} {r["M3"]["auc_entropy"]:8.3f}'
        if QUANTUM:
            line += (f' {r["M1"]["auc_s0"]:7.3f} {r["M1"]["gap_score"]:6.3f} {r["M1"]["gap_s0"]:7.3f} {r["M1"]["gap_reuploads"]:9.3f} '
                     f'{max(r["M1"]["decomposition_max_err"], r["M2"]["pauli_decomposition_max_err"]):8.1e} {r["E3"]["spearman_gapDelta_vs_dauc"]:7.2f} '
                     f'{r["E1"]["M1"]["rel_d"]:8.3f} {r["E1"]["random"]["rel_d"]:9.3f} {r["E2"]["auc"]["M1"]:7.3f} {r["E2"]["auc"]["gradient"]:8.3f} {r["E2"]["auc"]["random"]:8.3f}')
        L.append(line)
    L.append(f'  {"MEAN":14s} {m(lambda r: r["roc_auc"]):6.3f} {m(lambda r: r["M3"]["auc_residual"]):8.3f} {m(lambda r: r["M3"]["auc_entropy"]):8.3f}' +
             (f' {m(lambda r: r["M1"]["auc_s0"]):7.3f} {m(lambda r: r["M1"]["gap_score"]):6.3f} {m(lambda r: r["M1"]["gap_s0"]):7.3f} {m(lambda r: r["M1"]["gap_reuploads"]):9.3f} {"":8s} '
              f'{m(lambda r: r["E3"]["spearman_gapDelta_vs_dauc"]):7.2f} {m(lambda r: r["E1"]["M1"]["rel_d"]):8.3f} {m(lambda r: r["E1"]["random"]["rel_d"]):9.3f} '
              f'{m(lambda r: r["E2"]["auc"]["M1"]):7.3f} {m(lambda r: r["E2"]["auc"]["gradient"]):8.3f} {m(lambda r: r["E2"]["auc"]["random"]):8.3f}' if QUANTUM else ''))
    if QUANTUM:
        L += ['', '  Drift-vs-known gap per PC (amplitude part | re-upload part), mean over splits:',
              '    ' + ' '.join(f'PC{j + 1}:{m(lambda r: r["M1"]["gap_per_pc_amp"][j]):+.3f}|{m(lambda r: r["M1"]["gap_per_pc_reup"][j]):+.3f}' for j in range(PCA_DIM)),
              '  Pauli gap by weight: ' + ', '.join(f'{k}: {m(lambda r: r["M2"]["gap_by_weight"][k]):+.3f}' for k in ('1-qubit', '2-qubit', '3-qubit')),
              '  Re-upload gap profile (mean over splits): ' + ' '.join(f'{m(lambda r: r["M1"]["gap_per_reupload"][u]):+.3f}' for u in range(24)),
              '', '  AUC = ROC-AUC of the drift score; AUC res / ent = of the novelty residual / profile entropy (M3); AUC s0 = of the',
              '  amplitude-encoding score alone; gap = mean(drifting) - mean(known); err = max error of the exact decompositions (M1, M2);',
              '  rho E3 = Spearman(gap of Delta_u, ROC-AUC drop when re-upload u is removed); relD = distance after replacing the top-k',
              '  features by x_ref / original (E1, CADE k); ins = mean known-family percentile along the insertion curve (E2, lower = better).']
    open(os.path.join(out_root, 'aggregate_summary.txt'), 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('split', help=f"0-{N_SPLITS - 1}, 'all', or 'aggregate'")
    ap.add_argument('--data-dir', default=DEFAULT_DATA); ap.add_argument('--prefix', default='drebin_new')
    ap.add_argument('--model-results', default=DEFAULT_MODEL_RESULTS); ap.add_argument('--cade-results', default=DEFAULT_CADE_RESULTS)
    ap.add_argument('--run', type=int, default=0); ap.add_argument('--n-samples', type=int, default=200)
    ap.add_argument('--feature-names-dir', default='../../drift_analysis_cade/feature_names')
    ap.add_argument('--results-dir', default='results')
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__)); absp = lambda p: p if (not p or os.path.isabs(p)) else os.path.normpath(os.path.join(here, p))
    if a.split == 'aggregate':
        aggregate(absp(a.results_dir)); return
    for s in (range(N_SPLITS) if a.split == 'all' else [int(a.split)]):
        explain_split(s, a.data_dir, a.prefix, absp(a.model_results), absp(a.cade_results), a.run, a.n_samples,
                      absp(a.results_dir), absp(a.feature_names_dir))
    if a.split == 'all':
        aggregate(absp(a.results_dir))


if __name__ == '__main__':
    main()
