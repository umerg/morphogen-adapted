#!/usr/bin/env python3
"""Checkpoint selection sweep: the convergence curve and the argmin in one pass.

`headline_excess_mmd_morpho` over the morphometric vector IS the selection criterion --
the same one dendrite_gen and SemlaFlow are judged on -- so sweeping it across a run's
checkpoints answers two questions at once:

  * has training converged?  (the curve; the answer to "your baseline was undertrained")
  * which checkpoint do we report?  (the argmin)

That is why there is no separate cloud-quality probe: a cloud-level statistic would be
cheaper but would not be the thing selection actually uses.

    python tools/selection_sweep.py --phase sample ...   # GPU job
    python tools/selection_sweep.py --phase score  ...   # CPU job, no GPU, no checkpoint

Split because the two halves want different hardware: sampling is 1000 sequential
denoising steps per batch on a GPU, while reconstruction is embarrassingly parallel pure
CPU. `--phase all` runs both, for local testing.

Resumable at candidate granularity in both phases: a candidate whose output is already on
disk is skipped, so an interrupted job restarts where it stopped.

SELECTION IS ON VAL. Test is never used to choose a checkpoint; it is scored once, at the
end, on the winner.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

MORPHOGEN = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------ checkpoints

def find_checkpoints(runs_dir: Path, every: int, only: str) -> list[tuple[int, Path]]:
    """Candidate (epoch, path) pairs, numerically ordered.

    `epoch_%d.pth` is NOT lexicographically ordered -- epoch_9 sorts after epoch_10 --
    so sorting by name would silently pick a nonsense grid. Sort on the parsed suffix.
    """
    pairs = []
    for f in runs_dir.glob('epoch_*.pth'):
        m = re.fullmatch(r'epoch_(\d+)\.pth', f.name)
        if m:
            pairs.append((int(m.group(1)), f))
    if not pairs:
        raise SystemExit(f'no epoch_*.pth under {runs_dir}')
    pairs.sort()

    if only:
        want = {int(x) for x in only.split(',') if x.strip()}
        sel = [p for p in pairs if p[0] in want]
        missing = want - {e for e, _ in sel}
        if missing:
            raise SystemExit(f'requested epochs not present: {sorted(missing)}')
        return sel

    # Always include the final checkpoint whatever the stride lands on: it is the one
    # a reader assumes was evaluated, and "we didn't sample that one" is not an answer.
    sel = pairs[::every]
    if pairs[-1] not in sel:
        sel.append(pairs[-1])
    return sel


def run(cmd: list[str], env: dict | None = None) -> None:
    print('  $ ' + ' '.join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=MORPHOGEN,
                       env={**os.environ, **(env or {})})
    if r.returncode != 0:
        raise SystemExit(f'FAILED (exit {r.returncode}): {" ".join(str(c) for c in cmd)}')


# ---------------------------------------------------------------------- phases

def phase_sample(args, cands: list[tuple[int, Path]]) -> None:
    for ep, ckpt in cands:
        gen = args.out_dir / f'ep{ep:06d}'
        if (gen / 'clouds' / 'clouds.json').is_file():
            print(f'[sample] epoch {ep}: clouds present, skipping')
            continue
        print(f'[sample] epoch {ep} -> {gen}', flush=True)
        run([sys.executable, '-u', 'morphology_gen.py', '--stage', 'sample',
             '--dataroot', args.dataroot, '--category', args.category,
             '--model', ckpt, '--generate_dir', gen,
             '--max_samples', args.n, '--bs', args.bs, '--num_classes', args.num_classes,
             # --gpu 0 is not optional: opt.gpu defaults to None, which is threaded to
             # p_sample_loop as `device`, leaving the diffusion state on the CPU and
             # round-tripping it to the GPU on every one of the 1000 steps.
             '--gpu', '0', '--workers', args.sample_workers],
            env={'CUDA_VISIBLE_DEVICES': args.cuda_device})


def mid_set(gen: Path) -> set[str]:
    doc = json.loads((gen / 'clouds' / 'clouds.json').read_text())
    return {str(r['mid']) for r in doc['neurons']}


def phase_score(args, cands: list[tuple[int, Path]]) -> None:
    dirs = [(ep, args.out_dir / f'ep{ep:06d}') for ep, _ in cands]
    have = [(ep, d) for ep, d in dirs if (d / 'clouds' / 'clouds.json').is_file()]
    if not have:
        raise SystemExit('no sampled candidates found; run --phase sample first')
    if len(have) != len(dirs):
        missing = [ep for ep, d in dirs if (ep, d) not in have]
        print(f'WARNING: {len(missing)} candidates not sampled yet: {missing[:8]}')

    # GUARD 1: every candidate must have generated for the SAME source neurons.
    # --max_samples takes a fixed prefix of a fixed dataloader order, so this should
    # hold; if it ever stops holding the sweep is comparing different neurons and
    # nothing downstream would reveal it.
    ref_ep, ref_dir = have[0]
    ref = mid_set(ref_dir)
    for ep, d in have[1:]:
        got = mid_set(d)
        if got != ref:
            raise SystemExit(
                f'epoch {ep} generated a different neuron set from epoch {ref_ep} '
                f'({len(got ^ ref)} differ). The sweep would not be comparing like '
                'with like -- re-sample with an identical --max_samples and dataroot.')
    print(f'all {len(have)} candidates cover the same {len(ref)} source neurons')

    # GUARD 2: one GT reference set, built once and reused. A per-candidate GT subset
    # would make the comparison between candidates meaningless.
    gt_dir = args.gt_subset or (args.out_dir / 'gt_subset')
    if not (gt_dir.is_dir() and any(gt_dir.glob('*.swc'))):
        run([sys.executable, 'tools/gt_subset.py',
             '--manifest', ref_dir / 'clouds' / 'clouds.json',
             '--gt-root', args.gt_root, '--out-dir', gt_dir])
    n_gt = len(list(gt_dir.glob('*.swc')))
    print(f'GT reference: {n_gt} neurons at {gt_dir}')

    rows = []
    for ep, gen in have:
        out_json = gen / 'dist.json'
        if not out_json.is_file():
            print(f'\n[score] epoch {ep}', flush=True)
            if not (gen / 'manifest.json').is_file():
                run([sys.executable, '-u', 'morphology_gen.py', '--stage', 'reconstruct',
                     '--generate_dir', gen, '--recon_workers', args.workers,
                     '--detect_radius', args.detect_radius, '--root_cap', args.root_cap,
                     '--gamma_seed', args.gamma_seed,
                     '--length_threshold', args.length_threshold])
            if not (gen / 'trees' / 'manifest.json').is_file():
                run([sys.executable, '-u', 'morphology_gen.py', '--stage', 'finalize',
                     '--generate_dir', gen, '--train_npy_root', args.train_npy_root,
                     '--scale_mode', args.scale_mode, '--scale_seed', args.scale_seed])
            run([sys.executable, 'tools/score_trees.py',
                 '--gt-dir', gt_dir, '--pred-dir', gen / 'trees',
                 '--train-dir', args.train_dir, '--so2-axis', '0', '1', '0',
                 '--floor-seed', args.floor_seed, '--output-json', out_json])
        else:
            print(f'[score] epoch {ep}: dist.json present, skipping')

        j = json.loads(out_json.read_text())
        d, f = j['dist'], j.get('floor', {})
        rows.append({'epoch': ep,
                     'mmd_morpho': d.get('mmd_morpho'),
                     'density_morpho': d.get('density_morpho'),
                     'coverage_morpho': d.get('coverage_morpho'),
                     'excess': d.get('headline_excess_mmd_morpho'),
                     'floor_mmd': f.get('mmd_morpho')})

    rows.sort(key=lambda r: r['epoch'])
    (args.out_dir / 'curve.json').write_text(json.dumps(
        {'n': n_gt, 'scale_mode': args.scale_mode, 'split': 'val',
         'gt_dir': str(gt_dir), 'rows': rows}, indent=1))

    print('\n=== selection curve (val, N=%d) ===' % n_gt)
    print('%8s %12s %12s %12s %12s' % ('epoch', 'excess', 'mmd', 'density', 'coverage'))
    for r in rows:
        print('%8d %12s %12s %12s %12s' % (
            r['epoch'],
            '%.6f' % r['excess'] if r['excess'] is not None else '-',
            '%.6f' % r['mmd_morpho'] if r['mmd_morpho'] is not None else '-',
            '%.4f' % r['density_morpho'] if r['density_morpho'] is not None else '-',
            '%.4f' % r['coverage_morpho'] if r['coverage_morpho'] is not None else '-'))

    good = [r for r in rows if r['excess'] is not None]
    if good:
        best = min(good, key=lambda r: r['excess'])
        span = max(r['excess'] for r in good) - min(r['excess'] for r in good)
        print('\nargmin: epoch %d  (excess %.6f)' % (best['epoch'], best['excess']))
        print('spread across candidates: %.6f' % span)
        # A curve that does not move cannot select. Say so rather than reporting a
        # winner that is an artefact of the last decimal place -- Arm S gave
        # coverage 0.027, so saturation on this baseline is a live possibility.
        if span < 0.01 * abs(best['excess'] or 1.0):
            print('  NOTE: the curve is flat (spread < 1% of the value). Selection '
                  'cannot discriminate here -- take the last checkpoint and report '
                  'the flat curve as a finding about the metric\'s dynamic range.')
        (args.out_dir / 'best.json').write_text(json.dumps(
            {'epoch': best['epoch'], 'excess': best['excess'], 'spread': span,
             'n': n_gt, 'split': 'val'}, indent=1))

    _plot(args.out_dir, rows)


def _plot(out_dir: Path, rows: list[dict]) -> None:
    """Optional: the figure is a convenience, the JSON is the artefact."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:                                   # noqa: BLE001
        print(f'(no plot: {type(e).__name__}: {e})')
        return
    ep = [r['epoch'] for r in rows]
    fig, ax = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    ax[0].plot(ep, [r['excess'] for r in rows], 'o-', color='#1f77b4')
    ax[0].set_ylabel('headline_excess_mmd_morpho'); ax[0].grid(alpha=.3)
    ax[0].set_title('checkpoint selection (val)')
    ax[1].plot(ep, [r['coverage_morpho'] for r in rows], 'o-', label='coverage')
    ax[1].plot(ep, [r['density_morpho'] for r in rows], 's-', label='density')
    ax[1].set_xlabel('epoch'); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out_dir / 'curve.png', dpi=150)
    print(f'wrote {out_dir / "curve.png"}')


# ------------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--phase', choices=('sample', 'score', 'all'), required=True)
    ap.add_argument('--runs-dir', type=Path, required=True, help='holds epoch_*.pth')
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--every', type=int, default=3, help='checkpoint stride')
    ap.add_argument('--epochs', default='', help='explicit comma-separated epochs; '
                                                 'overrides --every')
    ap.add_argument('--n', type=int, default=512,
                    help='samples per candidate. MUST be identical across candidates; '
                         'only its constancy matters, not its size.')
    ap.add_argument('--bs', type=int, default=50)
    ap.add_argument('--num-classes', dest='num_classes', type=int, default=1)
    ap.add_argument('--dataroot', type=Path, default=None)
    ap.add_argument('--category', default='neurons')
    ap.add_argument('--cuda-device', dest='cuda_device', default='0',
                    help='CUDA_VISIBLE_DEVICES for the sample phase; a single id gives '
                         'DataParallel its one-device fast path')
    ap.add_argument('--sample-workers', dest='sample_workers', type=int, default=1)
    # scoring
    ap.add_argument('--gt-root', type=Path, default=None,
                    help='full GT SWC dir for the split; the N-matched subset is built '
                         'from it once')
    ap.add_argument('--gt-subset', type=Path, default=None,
                    help='reuse an existing GT subset instead of building one')
    ap.add_argument('--train-dir', type=Path, default=None,
                    help='train SWC dir for the real-vs-real floor')
    ap.add_argument('--train-npy-root', dest='train_npy_root', type=Path, default=None,
                    help='bake SYNSET dir; its train/ sidecars supply p_hat(m)')
    ap.add_argument('--workers', type=int, default=8, help='reconstruction processes')
    ap.add_argument('--scale-mode', dest='scale_mode',
                    choices=('marginal', 'oracle'), default='marginal')
    ap.add_argument('--scale-seed', dest='scale_seed', type=int, default=0)
    ap.add_argument('--floor-seed', dest='floor_seed', type=int, default=0)
    # MorphoGen+ reconstruction settings (train-calibrated, confirmed on held-out val)
    ap.add_argument('--detect-radius', dest='detect_radius', type=float, default=0.30)
    ap.add_argument('--root-cap', dest='root_cap', type=int, default=23)
    ap.add_argument('--gamma-seed', dest='gamma_seed', type=float, default=0.40)
    ap.add_argument('--length-threshold', dest='length_threshold', type=float, default=0.30)
    args = ap.parse_args()

    need = {'sample': ['dataroot'], 'score': ['gt_root', 'train_dir', 'train_npy_root']}
    for ph in (('sample', 'score') if args.phase == 'all' else (args.phase,)):
        for k in need.get(ph, []):
            if getattr(args, k) is None:
                raise SystemExit(f'--phase {ph} requires --{k.replace("_", "-")}')

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cands = find_checkpoints(args.runs_dir, args.every, args.epochs)
    print('%d candidates: %s' % (len(cands), [e for e, _ in cands]))

    if args.phase in ('sample', 'all'):
        phase_sample(args, cands)
    if args.phase in ('score', 'all'):
        phase_score(args, cands)


if __name__ == '__main__':
    main()
