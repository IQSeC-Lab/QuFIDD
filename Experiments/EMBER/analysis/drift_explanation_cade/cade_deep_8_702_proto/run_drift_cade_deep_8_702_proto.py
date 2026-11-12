#!/usr/bin/env python3
"""
run_drift_cade_deep_8_702_proto.py -- CADE-style drift explanation analysis (Yang et al., USENIX Security 2021, Sec. 5), for
the CADE-mini, deep (8-23-16-7); EMBER
===============================================================================================================
Standalone PyTorch port of CADE's released explanation code (github.com/whyisyoung/CADE; fork S-R-Nowmi/CADE):
cade/explain_by_distance.py + cade/mask_exp_by_distance_mask_m1.py (CADE), cade/explain_global_approximation_loose_
boundary.py + cade/mask_exp_by_approximation.py (boundary-based baseline), evaluate_explanation_by_distance.py (random,
gradient, metrics), and of COIN's released code (github.com/ninghaohello/Contextual-Outlier-Interpreter, src/
outlier_interpreter.py + prediction_strength.py, default hyper-parameters, as CADE used it). The trained PCA-8 model of
one task is re-loaded (PCA refitted on the same training split and checked against the saved scores).

Samples explained (as CADE): the test samples the analyst inspects, i.e. the top of the model's ranking up to the
stopping point of the best ranked F1 ("best inspection count"); CADE ranks the MAD-flagged samples, we rank all test
samples by the raw score (T = 0, our evaluation protocol). This list contains some known-family samples, as in CADE.
With --n-samples N > 0 a seeded random subset of N inspected samples is explained (the quantum model is costly).
Closest family y_t = argmin of the model's distance; x_ref = the training sample of y_t closest to its anchor.
Boundary of family c (CADE): lowerbound_c = median_c + 3.5 * MAD_c of the training distances of family c to its anchor.

Methods (k = number of features of the CADE explanation of the same sample):
  cade      distance-based mask m*m1 (m1 = features where x_t != x_ref): p ~ U(0,1), clipped to [0,1], Concrete
            relaxation (T 0.1), batch = x_t + 50 copies with one random feature flipped (1 - x), batch size 10,
            loss = mean ||f(x_t*(1-m*m1) + x_ref*m*m1) - c|| + lam*(||p*m1||_1 + ||p*m1||_2), Adam lr 0.01, 250
            epochs, lam_1 = 1e-3 adapted x/÷1.5 every 20 epochs against lowerbound, early stop; features with
            (m1 * p_best) == 1 are the explanation.
  boundary  CADE's "approximation_loose": per closest family c an MLP g_c (latent-15-2, softmax, Adam lr 0.01, batch 32,
            30 epochs) separates in-boundary training samples of c from out-of-boundary ones plus the inspected test
            samples closest to c and their Gaussian-noise copies (sigma = MAD_c, relabelled by the MAD rule); then a
            mask on x (x*m vs x*(1-m)) is solved against g_c(f(x)) (loss BCE(x*m) - BCE(x*(1-m)) + lam*EN(m), lr
            0.01, 250 epochs); features ranked by x*m, top min(#(x*m>0), k) + 1 (as in the released code).
  coin      COIN: context = 10% nearest in-distribution (training) samples, K = min(K_ps + 1, 4) KMeans clusters
            (prediction strength, 10 folds, threshold 0.85), per cluster (>= 5 samples) an L1 LinearSVC (C 1,
            intercept scaling 10) between the cluster and synthetic outliers (convex combinations shrunk towards x_t,
            rounded for binary data); importance = cluster-size-weighted |w|; top-k.
  random    k random features out of ALL features, 100 repetitions (as in the released code).
  gradient  (in the released code, not in the paper) rank |d sum(f(x) - c) / dx| and take the top-k features with a
            positive gradient; for fidelity models the gradient of the distance to the closest prototype.
Perturbation for the metrics (as CADE): binary data (Drebin) -> flip the selected features (0 <-> 1); continuous data
(EMBER, like CADE's IDS2018) -> replace them by the value of x_ref.
Metrics (CADE Tables 4 and 6): distance of the original / perturbed sample to the anchor of y_t (mean +- std), and the
success ratio = share of perturbed samples with distance <= lowerbound_{y_t}. Extra: the same with the FPR-10%
boundary (score exceeded by 10% of the known-family test samples), since MAD bounds are not comparable across geometries.

Model-specific choices (not in CADE): for the quantum model and the twins, f(x) is the state, the anchor its pure-state
prototype and the distance the fidelity distance 1 - |<v|psi>|^2; the boundary approximator g sees the vector of
fidelities to the C prototypes (for the CADE-minis, the latent vector itself).

Usage (from this folder):
  python3 run_drift_cade_deep_8_702_proto.py <split>      explain one split (seed run --run, default 0)
  python3 run_drift_cade_deep_8_702_proto.py aggregate    summary over the explained splits (../compare_drift.py: all models)
Outputs: results/split<s>/{explanations.csv, summary.json, selected_features.npz, case_study.txt}
"""
import os, sys, csv, json, time, argparse, warnings
import numpy as np

# ----------------------------------------------------------------------------- model of this folder
MODEL_KIND = 'cade'          # quantum | quantum_fixed | twin_c | twin_p | twin_r | cade
MODEL_TAG = 'cade_deep_8_702_proto'
DEFAULT_MODEL_RESULTS = '../../cade_pca8_p702_proto/cade_deep_8_702_proto/tau_1/results'
DEFAULT_DATA = '/home/snowm001/projects/OOD/Data2/EMBER/cade_ember2018_cadesplit_1000_k8'
DATASET = 'EMBER'          # EMBER | Drebin
BINARY = DATASET == 'Drebin'     # flip perturbation (binary features) vs. substitution with x_ref (continuous)
N_SPLITS = 8
PCA_DIM = 8
# CADE (explain_by_distance.py / mask_exp_by_distance_mask_m1.py)
EXP_LAMBDA_1, EXP_LR, EXP_EPOCH, EXP_TEMP = 1e-3, 1e-2, 250, 0.1
EXP_BATCH, EXP_NUM_SYNC, EXP_DISPLAY = 10, 50, 10
EXP_LAMBDA_PATIENCE, EXP_EARLY_STOP, EXP_MULT, EXP_ITER_THR = 20, 10, 1.5, 1e-4
MAD_T, MAD_B = 3.5, 1.4826
# boundary-based (explain_global_approximation_loose_boundary.py / mask_exp_by_approximation.py)
APX_HIDDEN, APX_LR, APX_BATCH, APX_EPOCHS = 15, 0.01, 32, 30
APX_MASK_EPOCH, APX_LAMBDA_PATIENCE, APX_EARLY_STOP, APX_ACC_LB = 250, 20, 250, 0.8
# COIN defaults (src/main.py)
COIN_RATIO_NBR, COIN_AUG, COIN_MIN_CLUSTER, COIN_MAX_K, COIN_VAL_TIMES, COIN_C, COIN_THRE_PS = 0.1, 10.0, 5, 4, 10, 1.0, 0.85
RANDOM_TRY = 100
METHODS = ('random', 'boundary', 'coin', 'cade', 'gradient')
DREBIN_NAMES = ['FakeInstaller', 'DroidKungFu', 'Plankton', 'GinMaster', 'BaseBridge', 'Iconosys', 'Kmin', 'FakeDoc']  # CADE top8 order
EMBER_GROUPS = [('byte_histogram', 256), ('byte_entropy_histogram', 256), ('strings', 104), ('general', 10),
                ('header', 62), ('section', 255), ('imports', 1280), ('exports', 128), ('data_directories', 30)]
N_Q, L_Q = 3, 62
UP_Q = (2, 5, 7, 10, 12, 15, 17, 20, 22, 25, 28, 30, 33, 35, 38, 40, 43, 45, 48, 50, 53, 56, 58, 61)
GROUPS = ([(0, 1, 2)] * 24 if MODEL_KIND == 'quantum_fixed'
          else [tuple([p % 8 for p in range(72)][3 * u:3 * u + 3]) for u in range(24)])


# ----------------------------------------------------------------------------- data
def load_split(data_dir, prefix, s):
    d = np.load(os.path.join(os.path.expanduser(data_dir), f'{prefix}_{s}.npz'), allow_pickle=True)
    dense = lambda a: a.item().toarray().astype(np.float32) if a.shape == () else a.astype(np.float32)
    Xtr, ytr, Xte, yte = dense(d['X_train']), d['y_train'].astype(np.int64), dense(d['X_test']), d['y_test'].astype(np.int64)
    known = sorted(int(c) for c in np.unique(ytr)); remap = {c: i for i, c in enumerate(known)}
    if 'families' in d.files:
        fams = [str(x) for x in d['families']]
    else:
        fams = DREBIN_NAMES if max(int(yte.max()), int(ytr.max())) < len(DREBIN_NAMES) else None
    if 'heldout_family' in d.files:
        hf = d['heldout_family']; fam = str(hf.item()) if hf.shape == () else str(hf)
    else:
        miss = sorted(set(np.unique(yte).tolist()) - set(known))
        fam = fams[int(miss[0])] if (fams and len(miss) == 1) else 'label_' + '_'.join(map(str, miss))
    name_of = lambda c: fams[c] if fams else str(c)
    return {'Xtr': Xtr, 'ytr': np.array([remap[int(c)] for c in ytr], dtype=np.int64), 'Xte': Xte, 'yte': yte,
            'is_ood': (~np.isin(yte, known)).astype(int), 'family': fam,
            'known_names': {i: name_of(c) for c, i in remap.items()}, 'test_names': [name_of(int(c)) for c in yte],
            'feature_index': d['feature_index'] if 'feature_index' in d.files else None}


def feature_names(n_feat, split, feature_index, names_dir):
    """Drebin: CADE's drebin_new{s}_train_selected_features.txt (if given); EMBER: group[column]."""
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
    """PCA-8 fitted on the training split (as in training), as a differentiable torch map x -> z, amp, angles."""
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
        self.signs = np.asarray(signs, dtype=float)
        sg = torch.tensor(self.signs, dtype=torch.float64)
        self.W = self.W0 * sg[:, None]; self.mu = self.mu0 * sg

    def z(self, X):
        return (X - self.m) @ self.W.T

    def amp_ang(self, Z):
        import torch
        amp = Z / torch.linalg.vector_norm(Z, dim=1, keepdim=True).clamp(min=1e-8)
        return amp, torch.arctan((Z - self.mu) / self.scale)


def make_circuit():
    import pennylane as qml
    dev = qml.device('default.qubit', wires=N_Q)

    @qml.qnode(dev, interface='torch', diff_method='backprop')
    def circuit(amp, ang, W, a, b):
        qml.StatePrep(amp, wires=range(N_Q), normalize=True)
        u = 0
        for l in range(L_Q):
            if l in UP_Q:
                for k in range(N_Q):
                    qml.RY(a[u, k] * ang[:, GROUPS[u][k]] + b[u, k], wires=k)
                u += 1
            for k in range(N_Q):
                qml.RX(W[l, k, 0], wires=k); qml.RY(W[l, k, 1], wires=k); qml.RZ(W[l, k, 2], wires=k)
            for k in range(N_Q - 1):
                qml.CNOT(wires=[k, k + 1])
            qml.CNOT(wires=[N_Q - 1, 0])
        return qml.state()
    return circuit


def build_embedding(model_dir, inp):
    import torch
    P = torch.load(os.path.join(model_dir, 'model_weights.pt'), map_location='cpu', weights_only=False)
    if MODEL_KIND.startswith('quantum'):
        circuit = make_circuit(); W, a, b = (P[k].double() for k in ('W', 'a', 'b'))
        def embed(X):
            amp, ang = inp.amp_ang(inp.z(X))
            return torch.cat([circuit(amp[i:i + 1024], ang[i:i + 1024], W, a, b) for i in range(0, len(amp), 1024)])
        return embed, 'fidelity'
    if MODEL_KIND.startswith('twin'):
        W1, b1, W2, b2 = (P[k].double() for k in ('W1', 'b1', 'W2', 'b2'))
        def embed(X):
            amp, ang = inp.amp_ang(inp.z(X)); A = torch.cat([amp, ang], 1)
            g = torch.tanh(A @ W1.T + b1) @ W2.T + b2
            if MODEL_KIND == 'twin_c':
                zc = torch.complex(amp + g[:, :PCA_DIM], g[:, PCA_DIM:])
            elif MODEL_KIND == 'twin_p':
                zc = torch.complex(g[:, :PCA_DIM], g[:, PCA_DIM:])
            else:
                zc = torch.complex(amp + g, torch.zeros_like(g))
            return zc / torch.linalg.vector_norm(zc, dim=1, keepdim=True).clamp(min=1e-8)
        return embed, 'fidelity'
    layers = sorted({int(k.split('.')[1]) for k in P if k.startswith('0.') and k.endswith('.weight')})
    Ws = [(P[f'0.{i}.weight'].double(), P[f'0.{i}.bias'].double()) for i in layers]
    def embed(X):
        h = inp.z(X)
        for j, (Wl, bl) in enumerate(Ws):
            h = h @ Wl.T + bl
            if j < len(Ws) - 1:
                h = torch.relu(h)
        return h
    return embed, 'euclidean'


def family_anchors(Etr, ytr, geom):
    import torch
    A = []
    for c in range(len(np.unique(ytr))):
        E = Etr[torch.as_tensor(ytr) == c]
        if geom == 'fidelity':
            rho = E.T @ E.conj() / len(E); A.append(torch.linalg.eigh(rho)[1][:, -1].conj())
        else:
            A.append(E.mean(0))
    return torch.stack(A)


def dist_all(E, anchors, geom):
    if geom == 'fidelity':
        return 1.0 - (E @ anchors.T).abs() ** 2
    return ((E[:, None, :] - anchors[None]) ** 2).sum(-1).clamp(min=1e-12).sqrt()


def dist_to(E, anchors, geom, c):
    """Distance of every embedding to the anchor of its own target family c (tensor of indices)."""
    if geom == 'fidelity':      # anchors hold conj(prototype), as in dist_all: <v_c|psi> = sum_i anchors[c]_i psi_i
        return 1.0 - (E * anchors[c]).sum(-1).abs() ** 2
    return ((E - anchors[c]) ** 2).sum(-1).clamp(min=1e-12).sqrt()


def reload_model(data, model_dir):
    import torch, itertools
    Xtr = torch.tensor(data['Xtr'], dtype=torch.float64); Xte = torch.tensor(data['Xte'], dtype=torch.float64)
    ytr = data['ytr']
    saved = np.loadtxt(os.path.join(model_dir, 'ood_scores.csv'), delimiter=',', skiprows=1, usecols=0)

    def full_check(inp):
        embed, geom = build_embedding(model_dir, inp)
        with torch.no_grad():
            Etr = embed(Xtr); anchors = family_anchors(Etr, ytr, geom)
            Dtr = dist_all(Etr, anchors, geom).numpy(); Ete = embed(Xte); Dte = dist_all(Ete, anchors, geom).numpy()
        rel = float(np.max(np.abs(Dte.min(1) - saved)) / (np.max(np.abs(saved)) + 1e-12))
        return rel, embed, geom, anchors, Dtr, Dte, Etr, Ete

    tried = []
    for pca_dtype in (np.float32, np.float64):
        inp = TorchInput(data['Xtr'], pca_dtype)
        out = full_check(inp); tried.append(f'{np.dtype(pca_dtype).name} {out[0]:.2e}')
        if out[0] <= 1e-3:
            break
    if out[0] > 1e-3:
        rs = np.random.default_rng(123)
        te_s = np.sort(rs.choice(len(Xte), min(150, len(Xte)), replace=False))
        tr_s = np.concatenate([rs.choice(np.flatnonzero(ytr == c), min(150, int((ytr == c).sum())), replace=False)
                               for c in np.unique(ytr)])
        Xsub = torch.cat([Xtr[tr_s], Xte[te_s]]); ysub = ytr[tr_s]; ntr = len(tr_s)
        rank = lambda a: np.argsort(np.argsort(a))
        emb_s, geom_s = build_embedding(model_dir, inp); cand = []
        for sg in itertools.product((1.0, -1.0), repeat=PCA_DIM):
            inp.set_signs(sg)
            with torch.no_grad():
                E = emb_s(Xsub); A = family_anchors(E[:ntr], ysub, geom_s)
                sc = dist_all(E[ntr:], A, geom_s).numpy().min(1)
            cand.append((np.corrcoef(rank(sc), rank(saved[te_s]))[0, 1], sg))
        cand.sort(key=lambda t: -t[0])
        for rho, sg in cand[:6]:
            inp.set_signs(sg); out = full_check(inp)
            tried.append(f'signs {"".join("+" if v > 0 else "-" for v in sg)} (rho {rho:.3f}) {out[0]:.2e}')
            if out[0] <= 1e-3:
                break
    if out[0] > 1e-3:
        raise SystemExit(f'reloaded model does not reproduce the saved scores ({"; ".join(tried)}) -- stopping')
    return (inp, Xtr, Xte) + out[1:], out[0], tried


# ----------------------------------------------------------------------------- CADE: distance-based mask (m * m1)
def cade_masks(embed, anchors, geom, x, xr, m1, cfam, lowerbound, rng, torch, epochs=EXP_EPOCH):
    """Batched port of OptimizeExp.fit_local (mask_exp_by_distance_mask_m1.py); each sample has its own mask, lambda
    schedule and early stopping, exactly as if it were explained alone. Returns (N, F) masks m1 * p_best (or NaN rows)."""
    N, F = x.shape; R = 1 + EXP_NUM_SYNC; eps = float(np.finfo(float).eps)
    # x_t + 50 synthetic copies with one random feature flipped (1 - x), as in fit_local
    rows = x[:, None, :].repeat(1, R, 1).clone()
    j = torch.as_tensor(rng.integers(0, F, size=(N, EXP_NUM_SYNC)))
    ar = torch.arange(N)[:, None]
    rows[ar, torch.arange(1, R)[None, :], j] = 1.0 - x[ar, j]
    chunks = [(i, min(i + EXP_BATCH, R)) for i in range(0, R, EXP_BATCH)]
    p = torch.rand(N, F, dtype=torch.float64).requires_grad_(True)     # RandomUniform(0, 1) initializer
    opt = torch.optim.Adam([p], lr=EXP_LR)
    lam = torch.full((N,), EXP_LAMBDA_1, dtype=torch.float64)
    lb = torch.as_tensor(lowerbound[cfam], dtype=torch.float64)
    cf = torch.as_tensor(cfam)
    active = torch.ones(N, dtype=torch.bool)
    up = torch.zeros(N); down = torch.zeros(N); es = torch.zeros(N)
    loss_best = torch.full((N,), float('inf'), dtype=torch.float64); sp_best = loss_best.clone()
    loss_last = loss_best.clone(); sp_last = loss_best.clone()
    best = torch.full((N, F), float('nan'), dtype=torch.float64)

    def losses(a, b):
        pn = p.clamp(0.0, 1.0)
        u = torch.rand(N, b - a, F, dtype=torch.float64)
        logit = (torch.log(pn + eps) - torch.log(1 - pn + eps))[:, None, :] + torch.log(u) - torch.log(1 - u)
        mask = torch.sigmoid(logit / EXP_TEMP) * m1[:, None, :]
        xe = rows[:, a:b] * (1 - mask) + xr[:, None, :] * mask
        Ef = embed(xe.reshape(-1, F)); d = dist_to(Ef, anchors, geom, cf.repeat_interleave(b - a)).reshape(N, b - a)
        l_exp = d.mean(1); pm = pn * m1
        l_sp = pm.abs().sum(1) + pm.pow(2).sum(1).clamp(min=1e-30).sqrt()
        return l_exp, l_sp

    for step in range(epochs):
        if not active.any():
            break
        tot, texp, tsp = torch.zeros(N, dtype=torch.float64), torch.zeros(N, dtype=torch.float64), torch.zeros(N, dtype=torch.float64)
        for a, b in chunks:
            keep = p.detach().clone()
            l_exp, l_sp = losses(a, b)
            opt.zero_grad(); ((l_exp + lam * l_sp) * active).sum().backward(); opt.step()
            with torch.no_grad():
                p[~active] = keep[~active]                                   # finished samples stay frozen
                l_exp, l_sp = losses(a, b)                                    # losses after the step (fresh noise, as sess.run)
                tot += l_exp + lam * l_sp; texp += l_exp; tsp += l_sp
        loss, l_exp, l_sp = tot / len(chunks), texp / len(chunks), tsp / len(chunks)
        with torch.no_grad():
            below = l_exp <= lb
            up = torch.where(active & below, up + 1, up); down = torch.where(active & ~below, down + 1, down)
            lam = torch.where(up >= EXP_LAMBDA_PATIENCE, lam * EXP_MULT, lam); up = torch.where(up >= EXP_LAMBDA_PATIENCE, 0 * up, up)
            lam = torch.where(down >= EXP_LAMBDA_PATIENCE, lam / EXP_MULT, lam); down = torch.where(down >= EXP_LAMBDA_PATIENCE, 0 * down, down)
            es = torch.where(active & (((loss - loss_last).abs() < EXP_ITER_THR) | ((l_sp - sp_last).abs() < EXP_ITER_THR)), es + 1, es)
            stop = active & below & (es >= EXP_EARLY_STOP)
            pn = p.clamp(0.0, 1.0)
            best[stop] = pn[stop]; active = active & ~stop
            if (step + 1) % EXP_DISPLAY == 0:
                p[torch.isnan(p)] = 1e-16
                imp = active & ((loss_best > loss) | (sp_best > l_sp))
                loss_best = torch.where(imp, loss, loss_best); sp_best = torch.where(imp, l_sp, sp_best)
                best[imp] = pn[imp]
            loss_last = torch.where(active, loss, loss_last); sp_last = torch.where(active, l_sp, sp_last)
    return (m1 * best).detach()


# ----------------------------------------------------------------------------- boundary-based ("approximation_loose")
def boundary_masks(embed, anchors, geom, Xtr, ytr, Dtr, x, cfam, k, med, mad, rng, torch, given_g=None):
    import torch.nn as nn
    C = anchors.shape[0]

    def latent(E):
        return (1.0 - dist_all(E, anchors, geom)) if geom == 'fidelity' else E

    def d_c(L, c):                     # distance to family c computed from the latent (MAD rule of the detector)
        return (1.0 - L[:, c]) if geom == 'fidelity' else ((L - anchors[c]) ** 2).sum(-1).sqrt()

    def drift(L, c):
        return (d_c(L, c) - med[c]).abs() / mad[c] > MAD_T

    with torch.no_grad():
        Ltr = latent(embed(Xtr)).double(); Lx = latent(embed(x)).double()
    gs, info = ({int(c): g for c, g in given_g.items()}, {'given_g': True}) if given_g else ({}, {})
    for c in ([] if given_g else np.unique(cfam)):
        Lc = Ltr[torch.as_tensor(ytr == c)]
        dr = drift(Lc, c); z_in, z_out = Lc[~dr], Lc[dr]
        z_test = Lx[torch.as_tensor(cfam == c)]
        cnt = len(z_in) - len(z_test)
        # as in the released code: the drift class is the test drift latents (+ synthetic drift); the out-of-boundary
        # training samples z_out are computed but then overwritten (z_drift = vstack(test_z_drift, z_syn_drift))
        z_drift = z_test
        if cnt > 0:
            times = int(round(cnt / len(z_test)))
            if times > 0:
                syn = (z_test[:, None, :] + torch.as_tensor(rng.normal(0, mad[c], size=(len(z_test), times, z_test.shape[1])))).reshape(-1, z_test.shape[1])
                sd = drift(syn, c); z_in = torch.cat([z_in, syn[~sd]]); z_drift = torch.cat([z_test, syn[sd]])
        X = torch.cat([z_in, z_drift]).float(); y = torch.cat([torch.zeros(len(z_in)), torch.ones(len(z_drift))]).long()
        g = nn.Sequential(nn.Linear(X.shape[1], APX_HIDDEN), nn.ReLU(), nn.Linear(APX_HIDDEN, 2))
        for lay in (g[0], g[2]):                             # Keras defaults: glorot_uniform kernels, zero biases
            nn.init.xavier_uniform_(lay.weight); nn.init.zeros_(lay.bias)
        opt = torch.optim.Adam(g.parameters(), lr=APX_LR, eps=1e-7)   # Keras Adam epsilon
        Y = torch.nn.functional.one_hot(y, 2).float()
        best_acc, best_state = -1.0, None                    # ModelCheckpoint(monitor='acc', save_best_only=True)
        for _ in range(APX_EPOCHS):
            perm = torch.randperm(len(X)); hit = 0
            for i in range(0, len(X), APX_BATCH):
                jj = perm[i:i + APX_BATCH]
                out_ = g(X[jj]); opt.zero_grad(); nn.functional.binary_cross_entropy(torch.softmax(out_, 1), Y[jj]).backward(); opt.step()
                hit += int((out_.argmax(1) == y[jj]).sum())
            if hit / len(X) > best_acc:
                best_acc, best_state = hit / len(X), {kk_: v.clone() for kk_, v in g.state_dict().items()}
        g.load_state_dict(best_state)
        with torch.no_grad():
            acc = float((g(X).argmax(1) == y).float().mean())
        for q in g.parameters():
            q.requires_grad_(False)
        gs[int(c)] = g.double(); info[int(c)] = {'acc': acc, 'n_in': int(len(z_in)), 'n_drift': int(len(z_drift)),
                                                 'n_train_out_of_boundary_unused': int(len(z_out))}

    cf = torch.as_tensor(cfam)
    def prob(Xin):
        L = latent(embed(Xin)).double(); out = torch.zeros(len(Xin), 2, dtype=torch.float64)
        for c, g in gs.items():
            sel = cf == c
            if sel.any():
                out[sel] = torch.softmax(g(L[sel]), 1)
        return out
    with torch.no_grad():
        y0 = prob(x)
    ok = y0.argmax(1) == 1                                   # the released code does not explain samples g calls in-dist
    N, F = x.shape
    m = torch.rand(N, F, dtype=torch.float64).requires_grad_(True); opt = torch.optim.Adam([m], lr=APX_LR, eps=1e-8)
    lam = torch.full((N,), EXP_LAMBDA_1, dtype=torch.float64)
    up = torch.zeros(N); down = torch.zeros(N); es = torch.zeros(N)
    loss_best = torch.full((N,), float('inf'), dtype=torch.float64); sp_best = loss_best.clone()
    loss_last = loss_best.clone(); sp_last = loss_best.clone(); best = torch.full((N, F), float('nan'), dtype=torch.float64)
    bce = lambda p_: -(y0 * torch.log(p_.clamp(1e-7, 1 - 1e-7)) + (1 - y0) * torch.log((1 - p_).clamp(1e-7, 1 - 1e-7))).mean(1)
    def objective():
        mn = m.clamp(0.0, 1.0)
        pe, pr = prob(x * mn), prob(x * (1 - mn))
        sp = m.abs().sum(1) + m.pow(2).sum(1).clamp(min=1e-30).sqrt()        # elastic net on the raw mask (mask_reshaped)
        return bce(pe) - bce(pr) + lam * sp, sp, pe
    active = ok.clone()
    for step in range(APX_MASK_EPOCH):
        if not active.any():
            break
        keep = m.detach().clone()
        opt.zero_grad(); loss, sp, _ = objective(); (loss * active).sum().backward(); opt.step()
        with torch.no_grad():
            m[~active] = keep[~active]
            loss, sp, pe = objective()
            acc2 = (pe.argmax(1) == y0.argmax(1)).double()
            hi = acc2 >= APX_ACC_LB
            up = torch.where(active & hi, up + 1, up); down = torch.where(active & ~hi, down + 1, down)
            lam = torch.where(up >= APX_LAMBDA_PATIENCE, lam * EXP_MULT, lam); up = torch.where(up >= APX_LAMBDA_PATIENCE, 0 * up, up)
            lam = torch.where(down >= APX_LAMBDA_PATIENCE, lam / EXP_MULT, lam); down = torch.where(down >= APX_LAMBDA_PATIENCE, 0 * down, down)
            es = torch.where(active & (((loss - loss_last).abs() < EXP_ITER_THR) | ((sp - sp_last).abs() < EXP_ITER_THR)), es + 1, es)
            stop = active & (acc2 > APX_ACC_LB) & (es >= APX_EARLY_STOP)
            best[stop] = m.clamp(0.0, 1.0)[stop]; active = active & ~stop
            loss_last = torch.where(active, loss, loss_last); sp_last = torch.where(active, sp, sp_last)
            if (step + 1) % EXP_DISPLAY == 0:
                m[torch.isnan(m)] = 1e-16
                imp = active & ((loss_best > loss) | (sp_best > sp))
                loss_best = torch.where(imp, loss, loss_best); sp_best = torch.where(imp, sp, sp_best)
                best[imp] = m.clamp(0.0, 1.0)[imp]
    sel = []
    xn = x.numpy(); bn = best.numpy()
    for i in range(N):
        if not bool(ok[i]) or np.isnan(bn[i]).any():
            sel.append(None); continue
        prod = xn[i] * bn[i]
        valid = min(int((prod > 0).sum()), int(k[i]))
        sel.append(np.argsort(prod, kind='mergesort')[::-1][:valid + 1])   # "+1" as in evaluate_explanation_by_distance.py
    return sel, {'g': info, 'n_not_drift_by_g': int((~ok).sum())}


# ----------------------------------------------------------------------------- COIN (released code, default settings)
def _closest_center(P, centers):
    return ((P[:, None, :] - centers[None]) ** 2).sum(-1).argmin(1)


def _prediction_strength(data_test, test_labels, train_centers, c):
    cc_of = _closest_center(data_test, train_centers); ps = []
    for cc in range(c):
        idx = np.flatnonzero(test_labels == cc); n = len(idx)
        if n <= 1:
            ps.append(float('inf')); continue
        lab = cc_of[idx]; same = (lab[:, None] == lab[None, :]).sum() - n
        ps.append((same / 2.0) / (n * (n - 1) / 2.0))
    return min(ps)


def _optimal_k(data, num_fold, max_k, thre, rng):
    from sklearn.cluster import KMeans
    n = len(data); avg = np.zeros(max_k + 1)
    for _ in range(num_fold):
        tr = rng.choice(n, int(n * 0.5), replace=False); te = np.setdiff1d(np.arange(n), tr)
        for c in range(1, max_k + 1):
            a = KMeans(n_clusters=c, n_init=10).fit(data[tr]); b = KMeans(n_clusters=c, n_init=10).fit(data[te])
            avg[c] += _prediction_strength(data[te], b.labels_, a.cluster_centers_, c)
    avg /= num_fold
    return max(i for i, v in enumerate(avg) if v > thre)


def _synthetic_sampling(insts, otlr, int_flag, rng):
    from sklearn.neighbors import NearestNeighbors
    n0 = insts.shape[0] - 1
    coeff = rng.random((n0, insts.shape[0])); coeff /= coeff.sum(1, keepdims=True)
    md = NearestNeighbors(n_neighbors=1).fit(insts).kneighbors([otlr])[0][0, 0] / insts.shape[1]
    new = coeff @ (insts - otlr[None, :])
    new *= (0.2 * rng.random((n0, 1)) * md)
    new += otlr[None, :]
    if int_flag:
        new = np.round(new)
    return np.vstack([otlr, new])


def coin_importance(nbrs, Xnormal, xt, rng):
    from sklearn.cluster import KMeans
    from sklearn import svm
    otlr_nbrs = Xnormal[nbrs.kneighbors([xt])[1][0]]
    k_best = min(_optimal_k(otlr_nbrs, COIN_VAL_TIMES, COIN_MAX_K, COIN_THRE_PS, rng) + 1, COIN_MAX_K)
    lab = KMeans(n_clusters=k_best, random_state=0, n_init=10).fit(otlr_nbrs).labels_
    score, nums = np.zeros(len(xt)), []
    for c in range(k_best):
        ins = otlr_nbrs[lab == c]
        if len(ins) < COIN_MIN_CLUSTER:
            continue
        c0 = _synthetic_sampling(ins, xt, int(BINARY), rng)
        clf = svm.LinearSVC(penalty='l1', C=COIN_C, dual=False, intercept_scaling=COIN_AUG)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            clf.fit(np.vstack([c0, ins]), np.r_[np.zeros(len(c0)), np.ones(len(ins))])
        score += len(ins) * np.abs(clf.coef_[0]); nums.append(len(ins))
    if not nums or score.sum() == 0:
        return None, k_best
    score /= float(np.sum(nums)); score /= score.sum()
    return score, k_best


# ----------------------------------------------------------------------------- one split
def perturb(x, feats, xref):
    xn = x.copy()
    if feats is None:
        return None
    feats = np.asarray(feats, dtype=int)
    if BINARY:
        xn[feats] = np.where(x[feats] == 0, 1.0, 0.0)
    else:
        xn[feats] = xref[feats]
    return xn


def explain_split(split, data_dir, prefix, model_results, run_id, n_samples, out_root, names_dir, epochs=EXP_EPOCH, seed=0):
    import torch
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '4')))
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    data = load_split(data_dir, prefix, split)
    model_dir = os.path.join(model_results, f'run{run_id}', f'split{split}')
    (inp, Xtr, Xte, embed, geom, anchors, Dtr, Dte, Etr, Ete), rel, tried = reload_model(data, model_dir)
    ytr, is_ood = data['ytr'], data['is_ood']; raw = Dte.min(1); C = anchors.shape[0]
    with torch.no_grad():                                  # consistency guard: per-family distance == model score
        _chk = dist_to(Ete, anchors, geom, torch.as_tensor(Dte.argmin(1))).numpy()
    assert np.abs(_chk - raw).max() < 1e-6, f'dist_to inconsistent with the model score ({np.abs(_chk - raw).max():.2e})'
    print(f'{MODEL_TAG} split {split} (withheld {data["family"]}): reload rel. diff {rel:.2e} ({"; ".join(tried)})', flush=True)
    # family boundaries (CADE detect.py: MAD = 1.4826 * median |d - median|)
    med = np.array([np.median(Dtr[ytr == c, c]) for c in range(C)])
    mad = np.array([max(MAD_B * np.median(np.abs(Dtr[ytr == c, c] - med[c])), 1e-12) for c in range(C)])
    lowerbound = med + MAD_T * mad
    thr10 = float(np.quantile(raw[is_ood == 0], 0.9))

    # inspected list: ranking by the raw score up to the best ranked-F1 stopping point
    order = np.argsort(-raw, kind='stable'); o = is_ood[order]
    tp = np.cumsum(o); kk = np.arange(1, len(o) + 1); prec, rec = tp / kk, tp / max(o.sum(), 1)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec + 1e-12), 0); best_cnt = int(f1.argmax()) + 1
    inspected = order[:best_cnt]
    idx = inspected if not (n_samples and best_cnt > n_samples) else np.sort(rng.choice(inspected, n_samples, replace=False))
    cfam = Dte[idx].argmin(1)
    ref_idx = np.array([np.flatnonzero(ytr == c)[Dtr[ytr == c, c].argmin()] for c in range(C)])
    x = Xte[idx]; xr = Xtr[ref_idx[cfam]]; m1 = (x != xr).double()
    print(f'  best inspection count {best_cnt} (precision {prec[best_cnt - 1]:.3f}); explaining {len(idx)} samples', flush=True)
    t = {}

    t0 = time.time()
    Mc = cade_masks(embed, anchors, geom, x, xr, m1, cfam, lowerbound, rng, torch, epochs=epochs); t['cade'] = time.time() - t0
    sel = {'cade': [None if torch.isnan(Mc[i]).any() else np.flatnonzero(Mc[i].numpy() == 1) for i in range(len(idx))]}
    k = np.array([len(s) if s is not None else 0 for s in sel['cade']])

    # gradient (released code): |d sum(f(x) - c)/dx| ranked, positive gradients only, up to k features
    xg = x.clone().requires_grad_(True); cf = torch.as_tensor(cfam)
    if geom == 'euclidean':
        (embed(xg) - anchors[cf]).sum().backward()
    else:
        dist_to(embed(xg), anchors, geom, cf).sum().backward()
    G = xg.grad.numpy(); sel['gradient'] = []
    for i in range(len(idx)):
        o_ = np.argsort(np.abs(G[i]))[::-1]; pos = o_[G[i][o_] > 0][:k[i]]
        sel['gradient'].append(pos if sel['cade'][i] is not None else None)

    t0 = time.time()
    sel['boundary'], binfo = boundary_masks(embed, anchors, geom, Xtr, ytr, Dtr, x, cfam, k, med, mad, rng, torch)
    t['boundary'] = time.time() - t0

    t0 = time.time()
    from sklearn.neighbors import NearestNeighbors
    Xnormal = data['Xtr'].astype(np.float64)
    nbrs = NearestNeighbors(n_neighbors=int(COIN_RATIO_NBR * (len(Xnormal) + len(idx)))).fit(Xnormal)
    sel['coin'], Ks = [], []
    for i in range(len(idx)):
        sc, kb = coin_importance(nbrs, Xnormal, x[i].numpy(), rng); Ks.append(kb)
        sel['coin'].append(None if (sc is None or sel['cade'][i] is None) else np.argsort(-sc, kind='mergesort')[:k[i]])
    t['coin'] = time.time() - t0
    print(f'  mean k {k.mean():.1f}; boundary {binfo}; COIN K mean {np.mean(Ks):.2f}; time ' +
          ', '.join(f'{a} {b:.0f}s' for a, b in t.items()), flush=True)

    # metrics
    xn, xrn = x.numpy(), xr.numpy()
    with torch.no_grad():
        d0 = dist_to(embed(x), anchors, geom, cf).numpy()
    res = {}
    def evaluate(S):
        keep = [i for i in range(len(idx)) if S[i] is not None]
        if not keep:
            return None
        Xp = torch.tensor(np.stack([perturb(xn[i], S[i], xrn[i]) for i in keep]))
        with torch.no_grad():
            E = embed(Xp); dp = dist_to(E, anchors, geom, cf[keep]).numpy(); smin = dist_all(E, anchors, geom).numpy().min(1)
        return keep, dp, dp <= lowerbound[cfam[keep]], smin <= thr10
    for n in ('cade', 'boundary', 'coin', 'gradient'):
        r = evaluate(sel[n]); res[n] = r
    # random: k random features out of all features, 100 repetitions
    keep = [i for i in range(len(idx)) if sel['cade'][i] is not None]
    dps, sus, s10 = [], [], []
    for _ in range(RANDOM_TRY if keep else 0):
        S = [rng.choice(xn.shape[1], size=k[i], replace=False) if i in keep else None for i in range(len(idx))]
        r = evaluate(S); dps.append(r[1]); sus.append(r[2]); s10.append(r[3])
    res['random'] = (keep, np.concatenate(dps), np.concatenate(sus), np.concatenate(s10)) if keep else None

    # outputs
    out = os.path.join(out_root, f'split{split}'); os.makedirs(out, exist_ok=True)
    names = feature_names(xn.shape[1], split, data['feature_index'], names_dir)
    with open(os.path.join(out, 'explanations.csv'), 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['test_index', 'real_family', 'is_drift', 'closest_family', 'd_orig', 'lowerbound', 'k',
                                        'd_cade', 'd_boundary', 'd_coin', 'd_gradient', 'cade_features'])
        pos = {n: ({i: j for j, i in enumerate(res[n][0])} if res[n] else {}) for n in ('cade', 'boundary', 'coin', 'gradient')}
        for i in range(len(idx)):
            dv = [f'{res[n][1][pos[n][i]]:.6g}' if i in pos[n] else '' for n in ('cade', 'boundary', 'coin', 'gradient')]
            w.writerow([int(idx[i]), data['test_names'][idx[i]], int(is_ood[idx[i]]), data['known_names'][int(cfam[i])],
                        f'{d0[i]:.6g}', f'{lowerbound[cfam[i]]:.6g}', int(k[i])] + dv +
                       ['|'.join(names[j] for j in sel['cade'][i]) if sel['cade'][i] is not None else ''])
    np.savez_compressed(os.path.join(out, 'selected_features.npz'), test_index=idx, closest=cfam,
                        **{n: np.array([np.asarray(s, int) if s is not None else np.array([-1]) for s in sel[n]], dtype=object)
                           for n in ('cade', 'boundary', 'coin', 'gradient')})
    with open(os.path.join(out, 'case_study.txt'), 'w') as fh:       # CADE Table 5 style, first explained drifting samples
        drift_pos = [i for i in range(len(idx)) if is_ood[idx[i]] == 1 and sel['cade'][i] is not None][:5]
        Xte_np, Xtr_np = data['Xte'], data['Xtr']
        real_mask = np.array(data['test_names']) == data['family']
        for i in drift_pos:
            c = int(cfam[i]); fam_tr = Xtr_np[ytr == c]
            fh.write(f'# test sample {int(idx[i])} (family {data["family"]}), closest family {data["known_names"][c]}, '
                     f'{k[i]} features\nfeature,value,avg_in_real_family_test,avg_in_closest_family_train,closest_sample_value\n')
            for j in sel['cade'][i]:
                fh.write(f'{names[j]},{xn[i, j]:g},{Xte_np[real_mask, j].mean():.4g},{fam_tr[:, j].mean():.4g},{xrn[i, j]:g}\n')
            fh.write('\n')
    summ = {'model': MODEL_TAG, 'dataset': DATASET, 'split': split, 'run': run_id, 'withheld_family': data['family'],
            'best_inspection_count': best_cnt, 'precision_at_stop': float(prec[best_cnt - 1]), 'n_explained': int(len(idx)),
            'n_explained_drift': int(is_ood[idx].sum()), 'n_features': int(xn.shape[1]), 'reload_rel_diff': rel,
            'k_mean': float(k[k > 0].mean()) if (k > 0).any() else 0.0, 'k_std': float(k[k > 0].std()) if (k > 0).any() else 0.0,
            'd_orig_mean': float(d0.mean()), 'd_orig_std': float(d0.std()), 'boundary': binfo,
            'coin_K_mean': float(np.mean(Ks)), 'time_s': t}
    for n in METHODS:
        r = res[n]
        if r is None:
            summ[n] = None; continue
        keep_, dp, su, s10_ = r
        summ[n] = {'n': int(len(keep_)), 'd_mean': float(dp.mean()), 'd_std': float(dp.std()),
                   'd_orig_mean': float(d0[keep_].mean()), 'rel_d_mean': float(dp.mean() / max(d0[keep_].mean(), 1e-12)),
                   'success_mad': float(su.mean()), 'success_fpr10': float(s10_.mean())}
    json.dump(summ, open(os.path.join(out, 'summary.json'), 'w'), indent=2)
    print('  ' + '  '.join(f'{n}: d {summ[n]["d_mean"]:.4g} succ {summ[n]["success_mad"]:.1%}' for n in METHODS if summ[n]), flush=True)


def aggregate(out_root):
    rows = [json.load(open(os.path.join(out_root, f'split{s}', 'summary.json'))) for s in range(N_SPLITS)
            if os.path.exists(os.path.join(out_root, f'split{s}', 'summary.json'))]
    if not rows:
        print('no explained splits in', out_root); return
    g = lambda r, n, key, f='{:.3f}': (f.format(r[n][key]) if r.get(n) else '-')
    L = [f'{MODEL_TAG} ({DATASET}) -- CADE Sec. 5 explanation analysis, {len(rows)}/{N_SPLITS} splits', '',
         'Table 4: distance of the perturbed sample to the anchor of its closest family (mean +- std)',
         f'  {"withheld":14s} {"n":>5} {"k":>6} {"original":>18s} ' + ' '.join(f'{n:>18s}' for n in METHODS)]
    for r in rows:
        L.append(f'  {r["withheld_family"][:14]:14s} {r["n_explained"]:5d} {r["k_mean"]:6.1f} {r["d_orig_mean"]:8.4g} +- {r["d_orig_std"]:<6.3g} ' +
                 ' '.join((f'{r[n]["d_mean"]:8.4g} +- {r[n]["d_std"]:<6.3g}' if r.get(n) else f'{"-":>18s}') for n in METHODS))
    L += ['', 'Table 6: share of perturbed samples that cross the detection boundary (distance <= median + 3.5 MAD of the closest family)',
          f'  {"withheld":14s} ' + ' '.join(f'{n:>9s}' for n in METHODS)]
    for r in rows:
        L.append(f'  {r["withheld_family"][:14]:14s} ' + ' '.join(f'{g(r, n, "success_mad", "{:.1%}"):>9s}' for n in METHODS))
    L += ['', 'Extra: share no longer flagged at the FPR-10% boundary',
          f'  {"withheld":14s} ' + ' '.join(f'{n:>9s}' for n in METHODS)]
    for r in rows:
        L.append(f'  {r["withheld_family"][:14]:14s} ' + ' '.join(f'{g(r, n, "success_fpr10", "{:.1%}"):>9s}' for n in METHODS))
    open(os.path.join(out_root, 'aggregate_summary.txt'), 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('split', help=f"0-{N_SPLITS - 1}, 'all', or 'aggregate'")
    ap.add_argument('--data-dir', default=DEFAULT_DATA); ap.add_argument('--prefix', default='drebin_new')
    ap.add_argument('--model-results', default=DEFAULT_MODEL_RESULTS)
    ap.add_argument('--run', type=int, default=0, help='seed run of the trained model to explain (0, 1, 2)')
    ap.add_argument('--n-samples', type=int, default=200, help='inspected samples explained per split (0 = all, as CADE)')
    ap.add_argument('--feature-names-dir', default='../feature_names', help='Drebin: folder with CADE feature-name files')
    ap.add_argument('--results-dir', default='results')
    ap.add_argument('--epochs', type=int, default=EXP_EPOCH, help='CADE mask epochs (250)')
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    absp = lambda p: p if os.path.isabs(p) else os.path.normpath(os.path.join(here, p))
    if a.split == 'aggregate':
        aggregate(absp(a.results_dir))
    else:
        for s in (range(N_SPLITS) if a.split == 'all' else [int(a.split)]):
            explain_split(s, a.data_dir, a.prefix, absp(a.model_results), a.run, a.n_samples, absp(a.results_dir),
                          absp(a.feature_names_dir), epochs=a.epochs)
        if a.split == 'all':
            aggregate(absp(a.results_dir))


if __name__ == '__main__':
    main()
