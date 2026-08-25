#!/usr/bin/env python3
"""Pin the scale contract that the reconstruction pipeline depends on.

MorphoGen reconstructs in unit-sphere space and restores scale afterwards
(tools/morphogen_swc.py). Four things have to hold for that to be safe, and
none of them is enforced by a type:

  1. the bake sidecars are readable and carry scale_m/centroid for every neuron
  2. restore_scale exactly inverts pc_normlize
  3. a missing sidecar fails LOUDLY, before any sampling happens
  4. clean_swc_tree is scale-equivariant -- which is what makes the position of
     the restore relative to the clean a convention rather than a bug

(4) is the one worth pinning hardest: if dendrite_gen ever gives clean_swc_tree
an absolute length scale, restoring after cleaning silently stops matching
restoring before, and every RECON-REF number shifts.

Usage
-----
  python tools/check_scale_contract.py                      # (4) only
  python tools/check_scale_contract.py --npy-root <out>/neurons --split val
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sub_process import pc_normlize  # noqa: E402
from tools.morphogen_swc import (  # noqa: E402
    df_to_graph, restore_scale, restore_scale_and_clean,
)

SCALES = (177.1, 1e4)       # a realistic scale_m, and four orders past it


def _synthetic_tree(seed=0):
    """A tree with a high-degree root and trifurcations, to exercise the splitter."""
    rng = np.random.default_rng(seed)
    rows = [[1, 1, 0.0, 0.0, 0.0, 1.0, -1]]
    for a in range(4):
        prev, ang = 1, 2 * np.pi * a / 4
        for st in range(6):                       # degree-2 chain -> collapse_degree2
            rows.append([len(rows) + 1, 3,
                         np.cos(ang) * (st + 1) * 3 + rng.normal(0, .2),
                         np.sin(ang) * (st + 1) * 3 + rng.normal(0, .2),
                         rng.normal(0, .2), 1.0, prev])
            prev = len(rows)
        for c in range(3):                        # trifurcation -> normalize_high_degree
            rows.append([len(rows) + 1, 3, np.cos(ang + .4 * c) * 22,
                         np.sin(ang + .4 * c) * 22, rng.normal(0, .2), 1.0, prev])
    return pd.DataFrame(rows, columns=['id', 'type', 'x', 'y', 'z', 'radius', 'parent'])


def check_equivariance(tol=1e-12):
    df = _synthetic_tree()
    ref = restore_scale_and_clean(df, 1.0, [0, 0, 0])
    refxyz = ref[['x', 'y', 'z']].to_numpy()
    ok = True
    for k in SCALES:
        out = restore_scale_and_clean(df, k, [0, 0, 0])
        topo = np.array_equal(out['parent'].to_numpy(), ref['parent'].to_numpy())
        err = np.abs(out[['x', 'y', 'z']].to_numpy() - refxyz * k).max() / (np.abs(refxyz).max() * k)
        good = topo and err < tol
        ok &= good
        print('  k=%-8g topology identical %-5s  max rel coord err %.1e   %s'
              % (k, topo, err, 'OK' if good else 'FAIL'))
    df_to_graph(ref)          # raises if the cleaned tree is not a single rooted tree
    return ok


def check_sidecars(npy_root, split):
    from morphology_gen import load_sidecars
    root = str(Path(npy_root).parent)
    synset = Path(npy_root).name
    npys = sorted(glob.glob(os.path.join(npy_root, split, '*.npy')))
    if not npys:
        print('  no .npy under %s/%s -- skipping' % (npy_root, split))
        return True
    mids = [(synset, '%s/%s' % (split, os.path.basename(f)[:-4])) for f in npys]

    sc = load_sidecars(root, mids)
    print('  join            : %d/%d sidecars loaded' % (len(sc), len(mids)))

    key, (m, c) = next(iter(sc.items()))
    raw = json.loads(open(os.path.join(root, key[0], key[1] + '.meta.json')).read())
    match = (m == raw['scale_m'] and c == raw['centroid'])
    print('  independent read: matches %s' % match)

    pts = np.load(os.path.join(root, key[0], key[1] + '.npy')).astype(np.float64)
    renorm, _, m2 = pc_normlize(restore_scale(pts, m, c))
    # .npy is float32, so exact equality is not available; 1e-6 relative is ample
    rt = np.abs(renorm - pts).max() < 1e-6 and abs(m2 - m) / m < 1e-6
    print('  round trip      : max|renorm-pts| %.1e | scale_m %.6f vs %.6f -> %s'
          % (np.abs(renorm - pts).max(), m2, m, 'OK' if rt else 'FAIL'))

    hidden = os.path.join(root, key[0], key[1] + '.meta.json')
    shutil.move(hidden, hidden + '.hidden')
    try:
        load_sidecars(root, mids)
        loud = False
    except SystemExit:
        loud = True
    finally:
        shutil.move(hidden + '.hidden', hidden)
    print('  fail-fast       : raises on a missing sidecar %s' % loud)
    return bool(match) and rt and loud


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npy-root', type=Path, default=None,
                    help='the SYNSET dir, e.g. <out>/neurons; omit to run the '
                         'equivariance check alone')
    ap.add_argument('--split', default='val')
    args = ap.parse_args()

    print('clean_swc_tree scale equivariance:')
    ok = check_equivariance()
    if args.npy_root:
        print('bake sidecars:')
        ok &= check_sidecars(args.npy_root, args.split)

    print('\n%s' % ('PASS: scale contract holds' if ok else 'FAIL: scale contract violated'))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
