import sys, numpy as np, torch
sys.argv=['x','--skip-cade']
exec(open('validate.py').read().split("# 2) our re-implementation")[0])
for seed in range(4):
    torch.manual_seed(seed); rng=np.random.default_rng(seed)
    selb, bi = P.boundary_masks(embed, A, 'euclidean', Xtr_t, ytr, Dtr, x, cl, k, med, mad, rng, torch)
    r = metric(selb); print(f'seed {seed}: our boundary port d {r[0]:.3f} +- {r[1]:.3f}; success {r[2]:.2%}; g acc', {c: round(v['acc'],3) for c,v in bi['g'].items()})
