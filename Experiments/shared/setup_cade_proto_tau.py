#!/usr/bin/env python3
"""setup_cade_proto_tau.py -- run ONCE from Experiments2 (login node is fine: it only copies and edits files).

Builds {dataset}/pca8/cade_pca8_p702_proto/<variant>_proto/tau_<t>/ for every CADE-mini variant and every
temperature t, mirroring the original cade_pca8_p702/<variant>/ folders (script + slurm, logs/ results/).

How the prototype loss gets into each script
  The encoder variant already has it: ablation_objective/cade_enc_8_702_proto/ vs cade_pca8_p702/cade_enc_8_702/.
  This script computes that change (insertions + the NAME line) and replays it into each variant's original
  script, placing every inserted block after the same context lines it follows in the encoder script.
  If a context is not found (or found more than once) in a variant, that variant is SKIPPED and reported --
  nothing is guessed. Each tau folder then gets its own copy with PROTO_TAU set to its value.

Usage:  python3 setup_cade_proto_tau.py [--taus 0.1 1 10] [--datasets EMBER Drebin lamda] [--dry-run]
"""
import argparse, difflib, os, re, shutil, sys

VARIANTS = ['cade_full_8_702', 'cade_enc_8_702', 'cade_tuned_8_702', 'cade_deep_8_702']
CTX = 3   # context lines used to anchor each inserted block


def read(p):
    with open(p) as fh:
        return fh.read().splitlines(keepends=True)


def norm(s):
    return s.strip()


def extract_patch(orig, proto):
    """Insert-only hunks of orig->proto, each anchored on the CTX non-blank orig lines before it,
    plus the line replacements (handled by rules below)."""
    sm = difflib.SequenceMatcher(a=orig, b=proto, autojunk=False)
    inserts, replaces = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'insert':
            ctx = [l for l in orig[:i1] if l.strip()][-CTX:]
            inserts.append((ctx, proto[j1:j2]))
        elif tag == 'replace':
            replaces.append((orig[i1:i2], proto[j1:j2]))
    return inserts, replaces


def find_anchor(lines, ctx):
    """Index after the last ctx line if the ctx sequence (non-blank lines, stripped) occurs exactly once."""
    nb = [(i, norm(l)) for i, l in enumerate(lines) if l.strip()]
    want = [norm(l) for l in ctx]
    hits = [k for k in range(len(nb) - len(want) + 1) if [x[1] for x in nb[k:k + len(want)]] == want]
    if len(hits) != 1:
        return None, len(hits)
    return nb[hits[0] + len(want) - 1][0] + 1, 1


def build_proto(variant, src_lines, inserts):
    out = list(src_lines)
    # apply inserts from the bottom up so earlier indices stay valid
    placed, notes = [], []
    for ctx, block in inserts:
        idx = None
        for k in range(len(ctx), 0, -1):          # full context first, then shorter, as long as it stays unique
            idx, n = find_anchor(out, ctx[-k:])
            if idx is not None:
                if k < len(ctx):
                    notes.append(f'block {block[0].strip()[:40]!r} placed after {ctx[-1].strip()[:50]!r} using {k}-line context')
                break
        if idx is None:
            return None, f'no unique anchor for inserted block {block[0].strip()[:60]!r} -> edit by hand'
        placed.append((idx, block))
    for idx, block in sorted(placed, key=lambda x: -x[0]):
        out[idx:idx] = block
    text = ''.join(out)
    # rename references to the script itself
    text = text.replace(f'{variant}.py', f'{variant}_proto.py')
    # NAME line: tag it, and define the two constants right after it
    m = re.search(r"^(NAME[ \t]*=[ \t]*['\"])(.*?)(['\"])[ \t]*$", text, flags=re.M)
    if not m:
        return None, 'no NAME = "..." line found'
    new = f"{m.group(1)}{m.group(2)} + prototype CE [ABLATION]{m.group(3)}\n" \
          f"PROTO_LAMBDA, PROTO_TAU = 1.0, 1.0   # weight and temperature of the Euclidean prototype loss"
    text = text[:m.start()] + new + text[m.end():]
    return text, notes


def set_tau(text, tau):
    text2, n = re.subn(r"^PROTO_LAMBDA,\s*PROTO_TAU\s*=\s*([0-9.eE+-]+),\s*[0-9.eE+-]+",
                       lambda m: f"PROTO_LAMBDA, PROTO_TAU = {m.group(1)}, {tau}", text, flags=re.M)
    if n != 1:
        raise RuntimeError('PROTO_TAU line not found exactly once')
    return text2


def make_slurm(src, variant, tau, ds):
    s = ''.join(read(src))
    s = s.replace(f'{variant}.py', f'{variant}_proto.py').replace(f'{variant}.slurm', f'{variant}_proto.slurm')
    tag = {'EMBER': 'em', 'Drebin': 'dr', 'lamda': 'la'}.get(ds, ds[:2].lower())
    short = variant.replace('cade_', '').replace('_8_702', '')
    s = re.sub(r'^#SBATCH --job-name=.*$', f'#SBATCH --job-name={tag}_p{short}_t{tau}', s, flags=re.M)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--taus', nargs='+', default=['0.1', '1', '10'])
    ap.add_argument('--datasets', nargs='+', default=['EMBER', 'Drebin', 'lamda'])
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    report, created = [], []
    for ds in a.datasets:
        base = f'{ds}/pca8'
        enc_orig = f'{base}/cade_pca8_p702/cade_enc_8_702/run_cade_enc_8_702.py'
        enc_prot = f'{base}/ablation_objective/cade_enc_8_702_proto/run_cade_enc_8_702_proto.py'
        if not (os.path.exists(enc_orig) and os.path.exists(enc_prot)):
            report.append(f'{ds}: missing {enc_orig} or {enc_prot} -> dataset skipped'); continue
        inserts, replaces = extract_patch(read(enc_orig), read(enc_prot))
        print(f'=== {ds}: encoder change = {len(inserts)} inserted blocks, {len(replaces)} replaced regions')
        for ctx, block in inserts:
            print(f'    insert {len(block)} line(s) after: {ctx[-1].strip()[:70]!r}')
        for variant in VARIANTS:
            src_dir = f'{base}/cade_pca8_p702/{variant}'
            src_py, src_sl = f'{src_dir}/run_{variant}.py', f'{src_dir}/run_{variant}.slurm'
            if not (os.path.exists(src_py) and os.path.exists(src_sl)):
                report.append(f'{ds}/{variant}: original script or slurm missing -> skipped'); continue
            if variant == 'cade_enc_8_702':
                text = ''.join(read(enc_prot))      # use the existing, already-run proto script as is
            else:
                text, info = build_proto(variant, read(src_py), inserts)
                if text is None:
                    report.append(f'{ds}/{variant}: {info} -> skipped'); continue
                for note in info:
                    report.append(f'{ds}/{variant}: CHECK {note}')
            for tau in a.taus:
                d = f'{base}/cade_pca8_p702_proto/{variant}_proto/tau_{tau}'
                py = f'{d}/run_{variant}_proto.py'; sl = f'{d}/run_{variant}_proto.slurm'
                if os.path.exists(py):
                    report.append(f'{d}: already exists -> left untouched'); continue
                if not a.dry_run:
                    os.makedirs(f'{d}/logs', exist_ok=True); os.makedirs(f'{d}/results', exist_ok=True)
                    with open(py, 'w') as fh: fh.write(set_tau(text, tau))
                    with open(sl, 'w') as fh: fh.write(make_slurm(src_sl, variant, tau, ds))
                created.append(d)
    print('\nCreated' + (' (dry run, nothing written)' if a.dry_run else '') + f': {len(created)} folders')
    for d in created:
        print('  ' + d)
    if report:
        print('\nNOT created / notes:')
        for r in report:
            print('  ' + r)
    if not a.dry_run and created:
        with open('submit_cade_proto_tau.sh', 'w') as fh:
            fh.write('#!/bin/bash\n# Submit every created tau folder (one 24-task array each). Run from Experiments2 in bash.\n')
            for d in created:
                fh.write(f'(cd {d} && sbatch run_{os.path.basename(os.path.dirname(d))}.slurm)\n')
        os.chmod('submit_cade_proto_tau.sh', 0o755)
        print('\nWrote submit_cade_proto_tau.sh -- check one folder first, then:  bash submit_cade_proto_tau.sh')


if __name__ == '__main__':
    sys.exit(main())
