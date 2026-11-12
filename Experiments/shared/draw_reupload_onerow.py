import csv, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
R = list(csv.DictReader(open('DX_reupload.csv')))
DS = [('EMBER','EMBER-2018'), ('Drebin','DREBIN'), ('LAMDA','LAMDA')]
MOD = [('Quantum (pair loss only)','-', r'$\mathrm{QuFiDD}_{\mathrm{pair}}$', 'pair'),
       ('Quantum (pair + prototype)','--', r'$\mathrm{QuFiDD}_{\mathrm{pair+proto}}$', 'pair+proto')]
def get(ds, mod, grp):
    r = sorted([x for x in R if x['dataset']==ds and x['model']==mod and x['group']==grp],
               key=lambda x: int(x['after_reupload']))
    return np.array([int(x['after_reupload']) for x in r]), np.array([float(x['mean_distance']) for x in r])
plt.rcParams.update({'font.size':7.5, 'pdf.fonttype':42})
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.0))
for j, (ds, dn) in enumerate(DS):
    a = ax[j]; txt = []
    for mod, ls, lab, short in MOD:
        x, dk = get(ds, mod, 'known'); _, dd = get(ds, mod, 'drifting')
        if ls == '-': a.fill_between(x, dk, dd, color='#2a78d6', alpha=0.10, lw=0)
        a.plot(x, dk, color='#808080', ls=ls, lw=1.2); a.plot(x, dd, color='#2a78d6', ls=ls, lw=1.2)
        txt.append('%s: $\\times$%.1f $\\rightarrow$ $\\times$%.1f' % (short, dd[0]/dk[0], dd[-1]/dk[-1]))
    a.text(0.97, 0.97, dn, transform=a.transAxes, ha='right', va='top', fontweight='bold')
    a.text(0.03, 0.04, 'drifting / known\n' + '\n'.join(txt), transform=a.transAxes, ha='left', va='bottom',
           fontsize=6.3, bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1.5))
    a.grid(alpha=0.35, lw=0.5); a.tick_params(labelsize=6.5, length=2, pad=1.5); a.set_xticks([0,5,10,15,20,25])
    for sp in ('top', 'right'): a.spines[sp].set_visible(False)
ax[0].set_ylabel('Mean distance to\nclosest prototype', fontsize=7)
ax[1].set_xlabel('Re-upload layer (0 = after encoding)', fontsize=7)
H = [Line2D([],[],color='#808080',ls='-',lw=1.2,label='known, '+MOD[0][2]),
     Line2D([],[],color='#2a78d6',ls='-',lw=1.2,label='drifting, '+MOD[0][2]),
     Line2D([],[],color='#808080',ls='--',lw=1.2,label='known, '+MOD[1][2]),
     Line2D([],[],color='#2a78d6',ls='--',lw=1.2,label='drifting, '+MOD[1][2])]
fig.legend(handles=H, loc='lower center', bbox_to_anchor=(0.5, 0.985), ncol=4, frameon=False,
           fontsize=7, handlelength=2.0, columnspacing=1.2, borderaxespad=0)
fig.subplots_adjust(left=0.075, right=0.995, top=0.97, bottom=0.17, wspace=0.17)
fig.savefig('figures/reupload_decomposition_onerow.pdf', bbox_inches='tight', pad_inches=0.01)
fig.savefig('figures/reupload_decomposition_onerow.png', bbox_inches='tight', pad_inches=0.01, dpi=200)
print('saved figures/reupload_decomposition_onerow.pdf')
