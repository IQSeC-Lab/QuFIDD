"""Redraw the failure-case t-SNE (QuFiDD pair+proto, seed 42) with the shared legend on the right.
usage: python draw_tsne_failure_right.py <coords.csv> [out_prefix]"""
import sys, csv, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
src = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else 'figures/fig_tsne_failure'
NAMES = {'EMBER': ['Xtrat','Zbot','Ramnit','Sality','Wapomi','Lethic','Ursnif','Kovter'],
         'Drebin': ['FakeInstaller','DroidKungFu','Plankton','GingerMaster','BaseBridge','Iconosys','Kmin','FakeDoc'],
         'lamda': ['DroidKungFu','Plankton','BaseBridge','SMSreg','Airpush','FakeInst','Shedun','SMSsend']}
PANELS = [('EMBER','Xtrat',72.5,True,'EMBER-2018'), ('Drebin','FakeInstaller',80.8,True,'DREBIN'),
          ('lamda','Plankton',52.6,False,'LAMDA')]
PAL = ['#2a78d6','#eb6834','#1baf7a','#eda100','#8e5bd0','#d6457a','#5a8f29']
rows = list(csv.DictReader(open(src)))
plt.rcParams.update({'font.size':6,'pdf.fonttype':42,'font.family':'STIXGeneral','mathtext.fontset':'stix'})
fig = plt.figure(figsize=(7.0,1.95))
gs = fig.add_gridspec(1,4,width_ratios=[1,1,1,0.72],wspace=0.07,left=0.005,right=0.995,top=0.92,bottom=0.17)
for i,(ds,unseen,auc,rings,dn) in enumerate(PANELS):
    ax = fig.add_subplot(gs[0,i])
    R = [r for r in rows if r['dataset']==ds and r['unseen']==unseen]
    xy = np.array([[float(r['x']),float(r['y'])] for r in R]); kind = np.array([r['kind'] for r in R])
    fam = np.array([r['family'] for r in R]); flag = np.array([r['flag_missed_or_fp']=='1' for r in R])
    kf = [f for f in NAMES[ds] if f!=unseen]; col = {f:PAL[j] for j,f in enumerate(kf)}
    m = kind=='train'; ax.scatter(*xy[m].T,s=1,c='#d8d8d8',lw=0,rasterized=True)
    for f in kf:
        m = (kind=='known')&(fam==f); ax.scatter(*xy[m].T,s=2.2,c=col[f],alpha=0.6,lw=0,rasterized=True)
    if rings:
        m = (kind=='known')&flag; ax.scatter(*xy[m].T,s=8,facecolors='none',edgecolors='black',linewidths=0.3,rasterized=True)
    m = (kind=='drift')&~flag; ax.scatter(*xy[m].T,s=4.5,c='black',marker='x',lw=0.35,rasterized=True)
    m = (kind=='drift')&flag;  ax.scatter(*xy[m].T,s=5.5,c='#d62728',marker='x',lw=0.5,rasterized=True)
    for f in kf:
        m = (kind=='proto')&(fam==f); ax.scatter(*xy[m].T,s=55,c=col[f],marker='*',edgecolors='black',linewidths=0.5,zorder=6)
    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values(): s_.set_color('#bbbbbb')
    ax.set_title(f'{dn}, unseen {unseen} (ROC-AUC {auc:.1f}%)',fontsize=6.5,pad=2)
    h = [Line2D([],[],marker='o',ls='none',ms=3,color=col[f],label=f) for f in kf]
    ax.legend(handles=h,loc='upper center',bbox_to_anchor=(0.5,0.0),ncol=4,frameon=False,fontsize=4.9,
              handletextpad=0.0,columnspacing=0.25,borderaxespad=0.2)
lax = fig.add_subplot(gs[0,3]); lax.axis('off')
common = [Line2D([],[],marker='x',ls='none',ms=4,color='black',label='Drifting, detected'),
          Line2D([],[],marker='x',ls='none',ms=4,color='#d62728',label='Drifting, missed\n(below median\nknown score)'),
          Line2D([],[],marker='o',ls='none',ms=4,markerfacecolor='none',markeredgecolor='black',label='Known, false positive\nat ranked-F1 stop'),
          Line2D([],[],marker='*',ls='none',ms=7,color='#888888',markeredgecolor='black',label='Prototype'),
          Line2D([],[],marker='o',ls='none',ms=3,color='#d8d8d8',label='Training')]
lax.legend(handles=common,loc='center left',frameon=False,fontsize=6,handletextpad=0.3,labelspacing=0.8,borderaxespad=0)
fig.savefig(out+'.pdf',dpi=300); fig.savefig(out+'.png',dpi=200); print('saved', out+'.pdf')
