#!/usr/bin/env python3
"""Bake raw SWC neurons into the fixed-size point clouds MorphoGen trains on.

Input : <raw-root>/{train,val,test}/<id>.swc   (degree-2 preserving, 0-indexed)
Output: <out-root>/<synset>/{train,val,test}/<id>.npy    exactly (15000, 3) float32
        <out-root>/<synset>/{train,val,test}/<id>.meta.json

Why the sidecar exists
----------------------
`pc_normlize` centres each neuron on its centroid and divides by its max radial
norm, so every training cloud is on the unit sphere and MorphoGen's training
distribution has *zero* scale variance by construction. Upstream's `makeNpy`
saved only the points, discarding the centroid and scale -- which makes every
scale-dependent evaluation metric (branch length, axial extent, radial span,
Sholl radius, path/radial-to-root) unrecoverable. We persist `centroid` and `m`
per neuron so scale can be restored after generation.

Layout
------
The loader (datasets/shapenet_data_pc.py) globs <root>/<synset>/<split>/*.npy and
derives the conditioning label from the synset index, so:

  --mode uncond  ->  one synset ("neurons")           : unconditional arm
  --mode class   ->  one synset per class ("class_0"..): class-conditional arm

Remember to register the synset names in `synsetid_to_cate`
(datasets/shapenet_data_pc.py) before training.

Usage
-----
  conda run -n MORPHOGEN python tools/swc_to_morphogen_npy.py \
      --raw-root  ~/Documents/neurons_raw \
      --label-root ~/Documents/neurons_conditional_full \
      --out-root  ~/Documents/neurons_morphogen_npy \
      --mode uncond --workers 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sub_process import farthest_point_sample_faster, pc_normlize, uni_sampling_up  # noqa: E402

NPOINTS = 15000          # datasets/shapenet_data_pc.py asserts exactly this
SPLITS = ('train', 'val', 'test')


def read_swc(path: Path) -> np.ndarray:
    """Return the raw 7-column SWC as float64, comments stripped."""
    arr = np.loadtxt(path, comments='#', ndmin=2)
    if arr.shape[1] < 7:
        raise ValueError(f'{path}: expected 7 SWC columns, got {arr.shape[1]}')
    return arr


def read_cell_class(path: Path):
    """Parse '# cell_class N' from a labelled SWC. Returns int or None."""
    if not path.is_file():
        return None
    with path.open() as f:
        for line in f:
            if not line.startswith('#'):
                break
            tok = line.lstrip('#').split()
            if len(tok) == 2 and tok[0] == 'cell_class':
                try:
                    return int(tok[1])
                except ValueError:
                    return None
    return None


def bake(swc: np.ndarray, npoints: int, seed: int):
    """SWC -> (points[npoints,3] on the unit sphere, centroid[3], scale m).

    Mirrors sub_process.py's __main__ pipeline, but seeded and returning the
    normalisation constants instead of throwing them away.
    """
    if len(swc) > npoints:
        _, pts = farthest_point_sample_faster(np.ascontiguousarray(swc[:, 2:5]), npoints, seed=seed)
    else:
        # Upsample by interpolating along branches, then FPS down to npoints.
        need = len(swc) + 3 * (npoints - len(swc))
        dense = uni_sampling_up(swc, need)
        _, pts = farthest_point_sample_faster(np.asarray(dense), npoints, seed=seed)

    pts, centroid, m = pc_normlize(np.asarray(pts, dtype=np.float64))
    return pts.astype(np.float32), np.asarray(centroid, dtype=np.float64), float(m)


def process_one(args):
    src, dst_npy, label_src, npoints, seed = args
    src, dst_npy = Path(src), Path(dst_npy)
    try:
        swc = read_swc(src)
        pts, centroid, m = bake(swc, npoints, seed)

        if pts.shape != (npoints, 3):
            raise ValueError(f'expected ({npoints}, 3), got {pts.shape}')
        if not np.isfinite(pts).all():
            raise ValueError('non-finite coordinates')

        dst_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(dst_npy, pts)
        meta = {
            'source': str(src),
            'n_swc_nodes': int(len(swc)),
            'centroid': centroid.tolist(),
            'scale_m': m,
            'cell_class': read_cell_class(Path(label_src)) if label_src else None,
            'seed': seed,
        }
        dst_npy.with_suffix('.meta.json').write_text(json.dumps(meta))
        return (str(src), None)
    except Exception as exc:  # keep going; report at the end
        return (str(src), f'{type(exc).__name__}: {exc}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--raw-root', type=Path, required=True,
                    help='corpus with degree-2 nodes: <root>/{train,val,test}/*.swc')
    ap.add_argument('--label-root', type=Path, default=None,
                    help='corpus carrying "# cell_class N" headers, joined by filename '
                         '(e.g. neurons_conditional_full). Omit for the unconditional arm.')
    ap.add_argument('--out-root', type=Path, required=True)
    ap.add_argument('--mode', choices=['uncond', 'class'], default='uncond',
                    help='uncond = one synset; class = one synset per cell class')
    ap.add_argument('--synset', default='neurons', help='synset name when --mode uncond')
    ap.add_argument('--npoints', type=int, default=NPOINTS)
    ap.add_argument('--splits', nargs='+', default=list(SPLITS))
    ap.add_argument('--limit', type=int, default=0, help='only the first N per split (smoke test)')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--seed', type=int, default=42, help='base seed; each neuron gets seed+index')
    args = ap.parse_args()

    jobs = []
    for split in args.splits:
        src_dir = args.raw_root / split
        if not src_dir.is_dir():
            sys.exit(f'missing split directory: {src_dir}')
        files = sorted(p for p in src_dir.iterdir()
                       if p.suffix == '.swc' and not p.name.startswith('._'))
        if args.limit:
            files = files[:args.limit]

        for i, f in enumerate(files):
            label_src = (args.label_root / split / f.name) if args.label_root else None
            if args.mode == 'class':
                cc = read_cell_class(label_src) if label_src else None
                if cc is None:
                    print(f'  skip (no cell_class): {f.name}')
                    continue
                synset = f'class_{cc}'
            else:
                synset = args.synset
            dst = args.out_root / synset / split / (f.stem + '.npy')
            jobs.append((str(f), str(dst), str(label_src) if label_src else '',
                         args.npoints, args.seed + i))

    print(f'baking {len(jobs)} neurons -> {args.out_root} '
          f'(mode={args.mode}, npoints={args.npoints}, workers={args.workers})')

    failures = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process_one, j) for j in jobs]
        for fut in as_completed(futs):
            src, err = fut.result()
            done += 1
            if err:
                failures.append((src, err))
            if done % 500 == 0 or done == len(jobs):
                print(f'  {done}/{len(jobs)}  failures={len(failures)}')

    print(f'\ndone: {done - len(failures)} written, {len(failures)} failed')
    for src, err in failures[:20]:
        print(f'  FAIL {src}: {err}')
    if failures:
        (args.out_root / 'failures.json').write_text(json.dumps(failures, indent=1))
        sys.exit(1)


if __name__ == '__main__':
    main()
