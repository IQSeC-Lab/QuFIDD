"""Validate the PyTorch port against CADE's released Drebin-FakeDoc model, masks and reported numbers (Tables 4/6)."""
import sys, time, csv, numpy as np, h5py, torch, importlib
sys.path.insert(0, '.'); P = importlib.import_module('drift_port')
import os; R = os.path.expanduser(os.environ.get('CADE_REPO', '~/CADE')); S = 7
d = np.load(f'{R}/data/drebin_new_{S}.npz'); Xtr, ytr0, Xte, yte0 = d['X_train'], d['y_train'], d['X_test'], d['y_test']
known = sorted(np.unique(ytr0)); mp = {c: i for i, c in enumerate(known)}
ytr = np.array([mp[c] for c in ytr0]); yte = np.array([mp.get(c, 7) for c in yte0])
f = h5py.File(f'{R}/models/drebin_new_{S}/cae_1340-512-128-32-7_lr0.0001_b64_e250_m10.0_lambda0.1_weights.h5', 'r')
Ws = [(torch.tensor(f[f'encoder_{i}/encoder_{i}/kernel:0'][()], dtype=torch.float64), torch.tensor(f[f'encoder_{i}/encoder_{i}/bias:0'][()], dtype=torch.float64)) for i in range(4)]
def embed(X):
    h = X
    for j, (W, b) in enumerate(Ws):
        h = h @ W + b
        if j < 3: h = torch.relu(h)
    return h
Xtr_t, Xte_t = torch.tensor(Xtr, dtype=torch.float64), torch.tensor(Xte, dtype=torch.float64)
info = np.load(f'{R}/reports/drebin_new_{S}/intermediate/mlp_training_info_for_detect_m10.0_lambda0.1.npz', allow_pickle=True)
with torch.no_grad():
    Etr = embed(Xtr_t); A = P.family_anchors(Etr, ytr, 'euclidean'); Dtr = P.dist_all(Etr, A, 'euclidean').numpy()
print('max |centroid - CADE centroid| =', float(np.abs(A.numpy() - np.array(info['centroids'])).max()))
C = 7; med = np.array([np.median(Dtr[ytr == c, c]) for c in range(C)])
mad = np.array([1.4826 * np.median(np.abs(Dtr[ytr == c, c] - med[c])) for c in range(C)])
print('max |MAD - CADE MAD| =', float(np.abs(mad - np.array(info['mad_family'])).max()))
lb = med + 3.5 * mad
rows = [l.strip().split(',') for l in open(f'{R}/reports/drebin_new_{S}/dist_mlp_one_by_one_check_pr_value_m10.0_mad3.5_lambda0.1.csv')][1:]
idx = np.array([int(r[0]) for r in rows[:127]]); cl = np.array([int(r[2]) for r in rows[:127]])
with torch.no_grad():
    Dte = P.dist_all(embed(Xte_t), A, 'euclidean').numpy()
print('closest family agrees with CADE list:', bool((Dte[idx].argmin(1) == cl).all()))
ref = np.array([np.flatnonzero(ytr == c)[Dtr[ytr == c, c].argmin()] for c in range(C)])
x = Xte_t[idx]; xr = Xtr_t[ref[cl]]; cf = torch.as_tensor(cl)
def metric(sel):
    keep = [i for i in range(len(idx)) if sel[i] is not None]
    Xp = torch.tensor(np.stack([P.perturb(x[i].numpy(), sel[i], xr[i].numpy()) for i in keep]))
    with torch.no_grad(): dp = P.dist_to(embed(Xp), A, 'euclidean', cf[keep]).numpy()
    return dp.mean(), dp.std(), (dp <= lb[cl[keep]]).mean(), len(keep)
with torch.no_grad(): d0 = P.dist_to(embed(x), A, 'euclidean', cf).numpy()
print(f'original distance {d0.mean():.3f} +- {d0.std():.3f}   (CADE report: 5.363 +- 0.568)')
# 1) CADE's own saved masks, our selection + perturbation + metric
mk = np.load(f'{R}/reports/drebin_new_{S}/mask_distance_mm1_0.001.npz', allow_pickle=True)['masks']
selc = [np.where(m == 1)[0] if m is not None else None for m in mk]; k = np.array([len(s) for s in selc])
print(f'[saved CADE masks]  k {k.mean():.3f} +- {k.std():.3f} (report 44.693 +- 6.151); d {metric(selc)[0]:.3f} +- {metric(selc)[1]:.3f} (0.065 +- 0.035); success {metric(selc)[2]:.2%} (97.64%)')
ma = np.load(f'{R}/reports/drebin_new_{S}/mask_approximation_loose_0.001.npz', allow_pickle=True)['masks']
sela = []
for i, m in enumerate(ma):
    prod = x[i].numpy() * m; v = min(int((prod > 0).sum()), int(k[i])); sela.append(np.argsort(prod, kind='mergesort')[::-1][:v + 1])
r = metric(sela); print(f'[saved boundary masks] d {r[0]:.3f} +- {r[1]:.3f} (3.960 +- 2.963); success {r[2]:.2%} (0.00%)')
rng = np.random.default_rng(0); dd = []; ss = []
for _ in range(100):
    sel = [rng.choice(x.shape[1], k[i], replace=False) for i in range(len(idx))]; r = metric(sel); dd.append(r[0]); ss.append(r[2])
print(f'[random x100] d {np.mean(dd):.3f} (5.423 +- 1.773); success {np.mean(ss):.2%} (0.00%)')
# 2) our re-implementation of the optimisations on CADE's model
torch.manual_seed(0); rng = np.random.default_rng(0)
if "--skip-cade" in sys.argv: sys.argv.append("--given-g")
if '--skip-cade' in sys.argv:
    selp=None
t = time.time(); m1 = (x != xr).double()
M = None if '--skip-cade' in sys.argv else P.cade_masks(embed, A, 'euclidean', x, xr, m1, cl, lb, rng, torch)
selp = selp if M is None else [None if torch.isnan(M[i]).any() else np.flatnonzero(M[i].numpy() == 1) for i in range(len(idx))]
if M is not None:
    kp = np.array([len(s) for s in selp if s is not None]); r = metric(selp)
    print(f'[our CADE port] n {r[3]} k {kp.mean():.1f} +- {kp.std():.1f}; d {r[0]:.3f} +- {r[1]:.3f}; success {r[2]:.2%}  ({time.time()-t:.0f}s)')
kk = k if M is None else np.array([len(s) if s is not None else 0 for s in selp])
t = time.time(); selb, bi = P.boundary_masks(embed, A, 'euclidean', Xtr_t, ytr, Dtr, x, cl, kk, med, mad, rng, torch)
r = metric(selb); print(f'[our boundary port] n {r[3]} d {r[0]:.3f} +- {r[1]:.3f}; success {r[2]:.2%} ({time.time()-t:.0f}s) {bi["n_not_drift_by_g"]} not drift by g')
if '--coin' in sys.argv:
    from sklearn.neighbors import NearestNeighbors
    t = time.time(); Xn = Xtr.astype(float); nb = NearestNeighbors(n_neighbors=int(0.1 * (len(Xn) + len(idx)))).fit(Xn); selco = []
    for i in range(len(idx)):
        sc, _ = P.coin_importance(nb, Xn, x[i].numpy(), rng); selco.append(None if sc is None else np.argsort(-sc, kind='mergesort')[:kk[i]])
    r = metric(selco); print(f'[our COIN port] n {r[3]} d {r[0]:.3f} +- {r[1]:.3f}; success {r[2]:.2%} (CADE: 6.219 +- 3.962, 0%) ({time.time()-t:.0f}s)')
if '--given-g' in sys.argv:
    import torch.nn as nn
    G = {}
    for c in np.unique(cl):
        hf = h5py.File(f'{R}/models/drebin_new_{S}/exp_approximation_loose/exp_mlp_family_{c}.h5', 'r'); h = hf['model_weights']
        g = nn.Sequential(nn.Linear(7, 15), nn.ReLU(), nn.Linear(15, 2)).double()
        with torch.no_grad():
            l0 = h['clf_0'][list(h['clf_0'].keys())[0]]; l1 = h['clf_1'][list(h['clf_1'].keys())[0]]
            g[0].weight.copy_(torch.tensor(l0['kernel:0'][()]).T); g[0].bias.copy_(torch.tensor(l0['bias:0'][()]))
            g[2].weight.copy_(torch.tensor(l1['kernel:0'][()]).T); g[2].bias.copy_(torch.tensor(l1['bias:0'][()]))
        for q in g.parameters(): q.requires_grad_(False)
        G[int(c)] = g
    kk = k  # CADE's own k list
    selg, bi = P.boundary_masks(embed, A, 'euclidean', Xtr_t, ytr, Dtr, x, cl, kk, med, mad, rng, torch, given_g=G)
    r = metric(selg); print(f'[our boundary mask with CADE g] n {r[3]} d {r[0]:.3f} +- {r[1]:.3f}; success {r[2]:.2%}; {bi["n_not_drift_by_g"]} not drift by g')
