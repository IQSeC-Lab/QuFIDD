#!/usr/bin/env python3
"""make_rank16.py -- run once from Experiments2:  python3 make_rank16.py
Writes rank_explainability16.py, a copy of rank_explainability.py that ranks all 16 models (pair-only and pair+proto
versions of every type) with the new names, and writes its outputs with the suffix _all16 (old files stay untouched)."""
import re
s = open('rank_explainability.py').read()
new = '''ORDER = ['quantum_hybrid_v7_p702_paironly', 'classical_fidtwin_c_702_paironly', 'classical_fidtwin_p_702_paironly',
         'classical_fidtwin_r_702_paironly', 'cade_full_8_702', 'cade_enc_8_702', 'cade_tuned_8_702', 'cade_deep_8_702',
         'quantum_hybrid_v7_p702', 'classical_fidtwin_c_702', 'classical_fidtwin_p_702', 'classical_fidtwin_r_702',
         'cade_full_8_702_proto', 'cade_enc_8_702_proto', 'cade_tuned_8_702_proto', 'cade_deep_8_702_proto']
NAMES = dict(zip(ORDER, ['QuFiDD_pair', 'QiCD^c_pair', 'QiCD^p_pair', 'QiCD^r_pair', 'CM^f_r+pair', 'CM^e_pair', 'CM^t_pair',
                         'CM^d_pair', 'QuFiDD_pair+proto', 'QiCD^c_pair+proto', 'QiCD^p_pair+proto', 'QiCD^r_pair+proto',
                         'CM^f_r+pair+proto', 'CM^e_pair+proto', 'CM^t_pair+proto', 'CM^d_pair+proto']))
TEX = dict(zip(ORDER, ['\\\\QF{pair}', '\\\\QC{c}{pair}', '\\\\QC{p}{pair}', '\\\\QC{r}{pair}', '\\\\CM{f}{r+pair}', '\\\\CM{e}{pair}',
                       '\\\\CM{t}{pair}', '\\\\CM{d}{pair}', '\\\\QF{pair+proto}', '\\\\QC{c}{pair+proto}', '\\\\QC{p}{pair+proto}',
                       '\\\\QC{r}{pair+proto}', '\\\\CM{f}{r+pair+proto}', '\\\\CM{e}{pair+proto}', '\\\\CM{t}{pair+proto}',
                       '\\\\CM{d}{pair+proto}']))
'''
s, n1 = re.subn(r"ORDER = \[.*?\n(?=METRICS = )", lambda m: new, s, flags=re.S)
s, n2 = re.subn(r"grp_break = \{.*?\}", "grp_break = {'classical_fidtwin_c_702_paironly': True, 'cade_full_8_702': True, "
                "'quantum_hybrid_v7_p702': True, 'classical_fidtwin_c_702': True, 'cade_full_8_702_proto': True}", s)
s, n3 = re.subn(r"sfx = f'_fpr", "sfx = '_all16' + f'_fpr", s)
assert (n1, n2, n3) == (1, 1, 1), (n1, n2, n3)
open('rank_explainability16.py', 'w').write(s)
print('wrote rank_explainability16.py')
