import os, glob, csv, numpy as np
M = [('Quantum (pair loss only)', 'ablation_objective/quantum_hybrid_v7_p702_paironly'),
     ('Quantum (pair + prototype)', 'quantum_pca8/pennylane/quantum_hybrid_v7_p702'),
     ('Twin, complex', 'twin_pca8_p702/classical_fidtwin_c_702'), ('Twin, plain', 'twin_pca8_p702/classical_fidtwin_p_702'),
     ('Twin, real', 'twin_pca8_p702/classical_fidtwin_r_702'), ('CADE-mini, full', 'cade_pca8_p702/cade_full_8_702'),
     ('CADE-mini, encoder', 'cade_pca8_p702/cade_enc_8_702'), ('CADE-mini, tuned', 'cade_pca8_p702/cade_tuned_8_702'),
     ('CADE-mini, deep', 'cade_pca8_p702/cade_deep_8_702'), ('PCA only, L2', 'pca_only/results_l2'), ('PCA only, raw', 'pca_only/results')]
B = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
rows = []
for ds in ['EMBER', 'Drebin', 'lamda']:
    print('=== %s: ranked F1 (%%) under an FPR budget on the known test samples (mean +- std over tasks)' % ds)
    print('%-28s %3s ' % ('model', 'n') + ' '.join('%11s' % ('<=%d%%' % round(b * 100)) for b in B))
    for lab, f in M:
        base = '%s/pca8/%s' % (ds, f) if f.startswith('pca_only') else '%s/pca8/%s/results' % (ds, f)
        F = {b: [] for b in B}; R = {b: [] for b in B}
        for p in sorted(glob.glob(base + '/run*/split*/ood_scores.csv')):
            with open(p) as fh:
                d = [(float(r['raw_score']), int(float(r['is_ood']))) for r in csv.DictReader(fh)]
            s = np.array([x[0] for x in d]); y = np.array([x[1] for x in d])
            o = y[np.argsort(-s, kind='stable')]; tp = np.cumsum(o); k = np.arange(1, len(o) + 1); fp = k - tp
            prec, rec = tp / k, tp / max(o.sum(), 1); f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0)
            nk = (1 - y).sum()
            for b in B:
                ok = fp <= b * nk + 1e-9
                if ok.any():
                    i = np.argmax(np.where(ok, f1, -1)); F[b].append(f1[i]); R[b].append(rec[i])
                else:
                    F[b].append(0.0); R[b].append(0.0)
        n = len(F[1.0])
        if n == 0:
            print('%-28s   0  (no ood_scores.csv under %s)' % (lab, base)); continue
        print('%-28s %3d ' % (lab, n) + ' '.join('%5.1f+-%4.1f' % (100 * np.mean(F[b]), 100 * np.std(F[b])) for b in B))
        for b in B:
            rows.append([ds, lab, n, round(b * 100), round(100 * np.mean(F[b]), 1), round(100 * np.std(F[b]), 1), round(100 * np.mean(R[b]), 1), round(100 * np.std(R[b]), 1)])
    print()
with open('FPR_BUDGET_ALL.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['dataset', 'model', 'n_tasks', 'fpr_budget_pct', 'rankedF1_mean', 'rankedF1_std', 'recall_mean', 'recall_std']); w.writerows(rows)
print('Budget 100% = no budget (= ranked F1 of the main table). Recall per budget is in FPR_BUDGET_ALL.csv')
