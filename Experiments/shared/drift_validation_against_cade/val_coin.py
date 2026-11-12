import sys, time, numpy as np, torch
sys.argv=['x','--skip-cade']
exec(open('validate.py').read().split("# 2) our re-implementation")[0])
from sklearn.neighbors import NearestNeighbors
rng=np.random.default_rng(0); t=time.time(); Xn=Xtr.astype(float)
nb=NearestNeighbors(n_neighbors=int(0.1*(len(Xn)+len(idx)))).fit(Xn); sel=[]; Ks=[]
for i in range(len(idx)):
    sc,kb=P.coin_importance(nb,Xn,x[i].numpy(),rng); Ks.append(kb); sel.append(None if sc is None else np.argsort(-sc,kind='mergesort')[:k[i]])
r=metric(sel); print(f'[our COIN port] n {r[3]} d {r[0]:.3f} +- {r[1]:.3f}; success {r[2]:.2%}; K mean {np.mean(Ks):.2f} (CADE: 6.219 +- 3.962, 0%) ({time.time()-t:.0f}s)')
