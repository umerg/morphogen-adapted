#!/usr/bin/env python3
"""Stage 2 -- reconstruction reference (RECON-REF) + wall-clock profile.

Pushes GROUND-TRUTH point clouds through the exact chain a generated cloud takes:

    baked .npy (15000)
      -> dataloader-style subsample to 2048        (must match, or the reference is
                                                    optimistically biased)
      -> L1_medial(NCenters=2048, iters=1)
      -> fps(1200)
      -> neuron_swc_generator
      -> filter_short_branches
      -> auxi (ResNet18 branch refinement)
      -> restore per-neuron scale
      -> clean_swc_tree (collapse deg-2 + binarise, same args as the corpus)

and scores the result against the GT neuron's own clean_swc_tree output. Any gap
here is the RECONSTRUCTION's cost, not the generative model's -- so it is the
reference every topological number for MorphoGen must be read against.

It is a *reference*, not a floor or a bound: a generated cloud can reconstruct into
a more GT-like tree than the GT's own cloud does.

Also reports per-stage wall-clock, which sets the budget for the full evaluation.

Usage
-----
  conda run -n MORPHOGEN python tools/recon_ref.py \
      --raw-root ~/Documents/neurons_raw --npy-root <out>/neurons \
      --split val --limit 20 --detect-radius 0.02 --root-cap 23
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

MORPHOGEN = Path(__file__).resolve().parent.parent
# dendrite_gen supplies the metric implementations; override for other checkouts.
DENDRITE_GEN = Path(os.environ.get('DENDRITE_GEN', Path.home() / 'Documents' / 'dendrite_gen'))
sys.path.insert(0, str(MORPHOGEN))
sys.path.insert(0, str(DENDRITE_GEN))

import torch  # noqa: E402

import morphology_gen as MG  # noqa: E402
from utils.cut import filter_short_branches  # noqa: E402
from utils.ske_connect import L1_medial, fps  # noqa: E402
from utils.swc_denoise import auxi  # noqa: E402

from preprocessing.clean_trees import clean_swc_tree  # noqa: E402
from validation.structural_metrics import (  # noqa: E402
    branch_order_values, mean_branch_length, partition_asymmetry, strahler_number,
)

# Shared with the generation path and the dendrite_gen adapter, so the scale
# restore cannot drift out of position. See tools/morphogen_swc.py.
from tools.morphogen_swc import (  # noqa: E402
    CLEAN_KW, SWC_COLS, df_to_graph, restore_scale_and_clean, to_one_indexed,
)


def summarise(G):
    root = G.graph['root']
    deg = dict(G.degree())
    return {
        'nodes': G.number_of_nodes(),
        'leaves': sum(1 for n in G.nodes if deg[n] == 1 and n != root),
        'root_degree': deg[root],
        'max_branch_order': float(np.max(branch_order_values(G, root=root))),
        'strahler': strahler_number(G, root=root),
        'partition_asymmetry': partition_asymmetry(G, root=root),
        'mean_branch_length': mean_branch_length(G),
    }


def reconstruct(cloud15k, meta, aux_model, args, rng, n_root_children=None, seed=None):
    """The generation-side chain, run on a GT cloud. Returns (graph, timings)."""
    t = {}
    # Match the dataloader exactly: test_points = last 5000, random subsample to npoints.
    te = cloud15k[10000:]
    pts = te[rng.choice(len(te), args.npoints)]

    t0 = time.time(); ske_L1 = L1_medial(points=pts, NCenters=2048, iters=1, seed=seed); t['L1'] = time.time() - t0
    t0 = time.time(); ske = ske_L1[fps(ske_L1, 1200, seed=args.seed), :];     t['fps'] = time.time() - t0
    t0 = time.time()
    swc = MG.neuron_swc_generator(ske, detect_radius=args.detect_radius,
                                  root_cap=args.root_cap, n_root_children=n_root_children,
                                  gamma_seed=args.gamma_seed, gamma_main=args.gamma_main,
                                  seed_direction=args.seed_direction)
    t['link'] = time.time() - t0

    t0 = time.time()
    cut = [{'n': int(r[0]), 'type': int(r[1]), 'x': r[2], 'y': r[3], 'z': r[4],
            'radius': r[5], 'parent': int(r[6])} for r in swc]
    cut = filter_short_branches(cut, length_threshold=args.length_threshold)
    nodes = auxi(pd.DataFrame(cut), aux_model)
    t['aux'] = time.time() - t0

    # Restore the per-neuron scale pc_normlize removed, then clean. Bundled so
    # the restore cannot be moved ahead of filter_short_branches/auxi, which are
    # calibrated in unit-sphere units.
    t0 = time.time()
    clean = restore_scale_and_clean(nodes, meta['scale_m'], meta['centroid'])
    t['clean'] = time.time() - t0
    return df_to_graph(clean), t


_WORKER = {}


def _init_worker(detect_radius, root_cap, length_threshold, npoints, seed, raw_root, split, k_from_gt,
                 gamma_seed, gamma_main, seed_direction):
    """Load the auxiliary CNN once per process."""
    torch.set_num_threads(1)          # we parallelise over neurons, not within them
    m = MG.ResNet18()
    m.load_state_dict(torch.load(MORPHOGEN / 'trained_model' / 'Auxiliary.pth', map_location='cpu'))
    m.eval()
    _WORKER.update(model=m, args=argparse.Namespace(
        detect_radius=detect_radius, root_cap=root_cap, length_threshold=length_threshold,
        npoints=npoints, seed=seed, k_from_gt=k_from_gt, gamma_seed=gamma_seed,
        gamma_main=gamma_main, seed_direction=seed_direction), raw_root=Path(raw_root), split=split)


def _run_one(npy_path):
    p = Path(npy_path)
    a = _WORKER['args']
    try:
        meta = json.loads(p.with_suffix('.meta.json').read_text())
        cloud = np.load(p).astype(np.float64)
        raw = pd.read_csv(_WORKER['raw_root'] / _WORKER['split'] / (p.stem + '.swc'),
                          comment='#', sep=r'\s+', header=None, names=SWC_COLS, usecols=range(7))
        gt = df_to_graph(clean_swc_tree(to_one_indexed(raw), **CLEAN_KW))
        # Per-neuron RNG so results do not depend on worker scheduling order.
        # NOT hash(): PYTHONHASHSEED randomises str hashing per process, which would make
        # this irreproducible across runs and inconsistent between workers.
        nseed = zlib.crc32(p.stem.encode()) ^ a.seed
        rng = np.random.default_rng(nseed)
        # Hand the reconstruction the GT primary-dendrite count, exactly as
        # dendrite_gen is handed num_root_children. Oracle input -- label it as such.
        k_gt = int(gt.degree(gt.graph['root'])) if a.k_from_gt else None
        rec, t = reconstruct(cloud, meta, _WORKER['model'], a, rng,
                             n_root_children=k_gt, seed=nseed)
        return {'file': p.stem, 'gt': summarise(gt), 'recon': summarise(rec),
                'k_gt': k_gt, 'cell_class': meta.get('cell_class'), 'time': t}, None
    except Exception as exc:
        return None, f'{p.stem}: {type(exc).__name__}: {exc}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw-root', type=Path, required=True)
    ap.add_argument('--npy-root', type=Path, required=True, help='the SYNSET dir, e.g. <out>/neurons')
    ap.add_argument('--split', default='val')
    ap.add_argument('--limit', type=int, default=20)
    ap.add_argument('--npoints', type=int, default=2048)
    ap.add_argument('--detect-radius', type=float, default=0.02)
    ap.add_argument('--root-cap', type=int, default=23)
    ap.add_argument('--length-threshold', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', type=Path, default=None, help='write per-neuron JSON here')
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--gamma-seed', type=float, default=1.0,
                    help='directional coefficient for SOMA edges (shipped: 1.0; paper states a '
                         'single gamma=1.2 for all edges)')
    ap.add_argument('--gamma-main', type=float, default=1.2)
    ap.add_argument('--seed-direction', choices=['cloud'], default=None,
                    help="estimate d_g from the skeleton's principal axis when pricing soma "
                         'edges. Shipped code leaves d_g at zero there, so soma edges never '
                         'receive the directional discount frontier edges get.')
    ap.add_argument('--k-from-gt', action='store_true',
                    help="seed the soma with the paired GT neuron's primary-dendrite count "
                         '(oracle input; dendrite_gen receives the same quantity)')
    ap.add_argument('--sample', type=int, default=0,
                    help='RANDOM sample of N neurons instead of the first N (--limit). '
                         'Taking the first N alphabetically is not class-representative.')
    args = ap.parse_args()

    all_npys = sorted((args.npy_root / args.split).glob('*.npy'))
    if args.sample:
        rs = np.random.default_rng(args.seed)
        idx = rs.choice(len(all_npys), min(args.sample, len(all_npys)), replace=False)
        npys = [all_npys[i] for i in sorted(idx)]
    else:
        npys = all_npys[:args.limit] if args.limit else all_npys

    init_args = (args.detect_radius, args.root_cap, args.length_threshold,
                 args.npoints, args.seed, str(args.raw_root), args.split, args.k_from_gt,
                 args.gamma_seed, args.gamma_main, args.seed_direction)

    rows, times, failures = [], Counter(), []
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.workers,
                                 initializer=_init_worker, initargs=init_args) as ex:
            for i, (row, err) in enumerate(ex.map(_run_one, [str(p) for p in npys], chunksize=4), 1):
                (failures.append(err) if err else rows.append(row))
                if i % 100 == 0 or i == len(npys):
                    print(f'  {i}/{len(npys)}  failures={len(failures)}', flush=True)
    else:
        _init_worker(*init_args)
        for i, p in enumerate(npys, 1):
            row, err = _run_one(str(p))
            (failures.append(err) if err else rows.append(row))
            if i % 100 == 0 or i == len(npys):
                print(f'  {i}/{len(npys)}  failures={len(failures)}', flush=True)

    for r in rows:
        for k, v in r.pop('time').items():
            times[k] += v

    if failures:
        print(f'\n{len(failures)} failures; first few:')
        for f in failures[:5]:
            print('  ', f)
    if not rows:
        sys.exit('no neurons succeeded')

    n = len(rows)
    keys = ['nodes', 'leaves', 'root_degree', 'max_branch_order', 'strahler',
            'partition_asymmetry', 'mean_branch_length']
    print(f'RECON-REF on {n} {args.split} neurons '
          f'(detect_radius={args.detect_radius}, root_cap={args.root_cap})\n')
    print(f'{"metric":22} {"GT median":>12} {"RECON median":>14} {"ratio":>8}')
    print('-' * 60)
    for k in keys:
        g = float(np.median([r['gt'][k] for r in rows]))
        c = float(np.median([r['recon'][k] for r in rows]))
        print(f'{k:22} {g:12.3f} {c:14.3f} {c / g if g else float("nan"):8.2f}x')

    tot = sum(times[k] for k in ('L1', 'fps', 'link', 'aux', 'clean'))
    print(f'\nwall-clock per neuron (n={n}):')
    for k in ('L1', 'fps', 'link', 'aux', 'clean'):
        print(f'  {k:10} {times[k] / n:7.3f}s  ({100 * times[k] / tot:4.1f}%)')
    print(f'  {"TOTAL":10} {tot / n:7.3f}s')
    for split, cnt in (('val', 2529), ('test', 1167)):
        hrs = tot / n * cnt / 3600
        print(f'  -> {split} ({cnt} neurons): {hrs:.2f} core-hours '
              f'= {hrs * 60 / 32:.1f} min across 32 workers')

    if args.out:
        args.out.write_text(json.dumps(rows, indent=1))
        print(f'\nper-neuron detail -> {args.out}')


if __name__ == '__main__':
    main()
