import csv, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
R = list(csv.DictReader(open('DX_reupload.csv')))
DS = [('EMBER','EMBER-2018'), ('Drebin','DREBIN'), ('LAMDA','LAMDA')]
MOD = [('Quantum (pair loss only)','-', r'$\mathrm{QuFiDD}_{\mathrm{pair}}$'),
       ('Quantum (pair + prototype)','--', r'$\mathrm{QuFiDD}_{\mathrm{pair+proto}}$')]
def get(ds, mod, grp):
    r = sorted([x for x in R if x['dataset']==ds and x['model']==mod and x['group']==grp],
               key=lambda x: int(x['after_reupload']))
    return np.array([int(x['after_reupload']) for x in r]), np.array([float(x['mean_distance']) for x in r])
PROTO_IN_DIST = False
COL = {'EMBER':'#8e5bd0', 'Drebin':'#e8a200', 'LAMDA':'#d6457a'}
plt.rcParams.update({'font.size':7, 'pdf.fonttype':42})
fig, (a0, a1) = plt.subplots(1, 2, figsize=(3.5, 2.2))
for ds, dn in DS:
    c = COL[ds]
    for mod, ls, lab in MOD:
        x, dk = get(ds, mod, 'known'); _, dd = get(ds, mod, 'drifting')
        if ls == '-' or PROTO_IN_DIST:
            al = 1.0 if ls == '-' else 0.45
            a0.plot(x, dd, color=c, ls='-' if ls == '-' else '--', lw=1.2, alpha=al)
            a0.plot(x, dk, color=c, ls=':', lw=1.2, alpha=al)
        r = dd / dk
        a1.plot(x, r, color=c, ls=ls, lw=1.2)
        if ls == '-':
            a1.text(x[-1] + 0.6, r[-1], dn, color=c, fontsize=6, va='center', ha='left', fontweight='bold')
for a in (a0, a1):
    a.grid(alpha=0.35, lw=0.5); a.tick_params(labelsize=6, length=2, pad=1.5)
    a.set_xticks([0, 6, 12, 18, 24]); a.set_xlabel('Re-upload layer', fontsize=6.5, labelpad=1)
    for sp in ('top', 'right'): a.spines[sp].set_visible(False)
a0.set_ylabel('Mean distance to prototype', fontsize=6.5, labelpad=1)
a1.set_ylabel('Drifting / known', fontsize=6.5, labelpad=1)
a1.set_xlim(right=31)
H = [Line2D([],[],color=COL[d],lw=1.5,label=n) for d, n in DS] + [
     Line2D([],[],color='black',ls='-',lw=1.1,label='Drifting'), Line2D([],[],color='black',ls=':',lw=1.1,label='Known'),
     Line2D([],[],color='black',ls='-',lw=1.1,label=MOD[0][2]), Line2D([],[],color='black',ls='--',lw=1.1,label=MOD[1][2])]
fig.legend(handles=H, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=4, frameon=False, fontsize=5.8,
           handlelength=1.8, columnspacing=0.8, handletextpad=0.4, borderaxespad=0)
fig.subplots_adjust(left=0.12, right=0.99, top=0.98, bottom=0.17, wspace=0.32)
fig.savefig('figures/reupload_decomposition_2panel.pdf', bbox_inches='tight', pad_inches=0.01)
fig.savefig('figures/reupload_decomposition_2panel.png', bbox_inches='tight', pad_inches=0.01, dpi=250)
print('saved figures/reupload_decomposition_2panel.pdf')
