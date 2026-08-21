#!/usr/bin/env python3
"""Stage 1 -- data-pipeline round trip.

Verifies that a neuron survives
    raw SWC -> sub_process bake -> .npy -> dataloader normalise -> un-normalise
               -> scale restore
without its geometry moving. If this does not hold, nothing downstream can be
trusted, so this runs before any GPU time is requested.

Scope: this stage is GEOMETRIC only. A point cloud carries no topology, so
tree-level agreement (branch counts, asymmetry, ...) cannot be checked here --
that needs the reconstruction chain and is Stage 2 (RECON-REF).

Extent statistics are computed with dendrite_gen's own
`validation/geometric_metric.py` so the numbers come from exactly the code that
will judge the baseline. Neurons use uhat = (0, 1, 0).

Usage
-----
  conda run -n MORPHOGEN python tools/roundtrip_check.py \
      --raw-root ~/Documents/neurons_raw --npy-root <out>/neurons \
      --split val --limit 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# dendrite_gen supplies the metric implementations; override for other checkouts.
DENDRITE_GEN = Path(os.environ.get('DENDRITE_GEN', Path.home() / 'Documents' / 'dendrite_gen'))
sys.path.insert(0, str(DENDRITE_GEN))
from validation.geometric_metric import bbox_diag_length, height_z_range  # noqa: E402

UHAT = (0.0, 1.0, 0.0)          # config/neuron_type_conditional_run.yaml: so2_axis


def radial_span(pts, uhat=UHAT):
    """Max distance from the centroid within the plane perpendicular to uhat.

    Same quantity as validation.geometric_metric.span_xy_diameter up to a factor
    of two, but O(N) instead of O(N^2) -- span_xy_diameter builds a full pairwise
    matrix, which is 15000^2 here.
    """
    pts = np.asarray(pts, float).reshape(-1, 3)
    u = np.asarray(uhat, float) / np.linalg.norm(uhat)
    perp = pts - np.outer(pts @ u, u)
    return float(np.linalg.norm(perp - perp.mean(0), axis=1).max())


def chamfer(a, b, sample=4000, seed=0):
    """Symmetric mean nearest-neighbour distance, subsampled for tractability."""
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)
    a = a[rng.choice(len(a), min(sample, len(a)), replace=False)]
    b = b[rng.choice(len(b), min(sample, len(b)), replace=False)]
    da, _ = cKDTree(b).query(a)
    db, _ = cKDTree(a).query(b)
    return float(da.mean() + db.mean())


def stats(pts):
    pts = np.asarray(pts, float).reshape(-1, 3)
    c = pts.mean(0)
    return {
        'centroid': c,
        'axial_extent': height_z_range(pts, UHAT),
        'radial_span': radial_span(pts),
        'bbox_diag': bbox_diag_length(pts),
        'max_radial': float(np.linalg.norm(pts - c, axis=1).max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw-root', type=Path, required=True)
    ap.add_argument('--npy-root', type=Path, required=True,
                    help='the SYNSET directory, e.g. <out>/neurons')
    ap.add_argument('--split', default='val')
    ap.add_argument('--limit', type=int, default=20)
    ap.add_argument('--tol', type=float, default=5.0, help='max allowed %% error')
    args = ap.parse_args()

    npys = sorted((args.npy_root / args.split).glob('*.npy'))[:args.limit]
    if not npys:
        sys.exit(f'no .npy under {args.npy_root / args.split}')

    # The dataloader standardises with a GLOBAL (per-axis mean, scalar std) computed
    # over the train split, then generation inverts it with x*s+m. Emulate that here
    # so the round trip covers the same arithmetic.
    clouds = [np.load(p).astype(np.float64) for p in npys]
    allpts = np.concatenate(clouds, 0)
    g_mean = allpts.mean(0, keepdims=True)
    g_std = allpts.reshape(-1).std()

    rows, worst = [], {}
    for p, cloud in zip(npys, clouds):
        meta = json.loads(p.with_suffix('.meta.json').read_text())
        raw = np.loadtxt(args.raw_root / args.split / (p.stem + '.swc'), comments='#', ndmin=2)
        raw_pts = raw[:, 2:5]

        # dataloader normalise -> un-normalise (must be an identity)
        norm = (cloud - g_mean) / g_std
        back = norm * g_std + g_mean
        identity_err = float(np.abs(back - cloud).max())

        # restore the per-neuron scale discarded by pc_normlize
        restored = back * meta['scale_m'] + np.asarray(meta['centroid'])

        s_raw, s_new = stats(raw_pts), stats(restored)
        row = {'file': p.stem, 'identity_err': identity_err,
               'centroid_shift': float(np.linalg.norm(s_raw['centroid'] - s_new['centroid'])),
               'chamfer_um': chamfer(raw_pts, restored)}
        for k in ('axial_extent', 'radial_span', 'bbox_diag', 'max_radial'):
            err = 100.0 * abs(s_new[k] - s_raw[k]) / max(s_raw[k], 1e-9)
            row[k + '_pct'] = err
            worst[k] = max(worst.get(k, 0.0), err)
        rows.append(row)

    print(f'{"neuron":30} {"axial%":>7} {"radial%":>8} {"bbox%":>7} {"maxrad%":>8} '
          f'{"cshift":>7} {"chamfer":>8} {"ident":>9}')
    for r in rows:
        print(f'{r["file"][:30]:30} {r["axial_extent_pct"]:7.2f} {r["radial_span_pct"]:8.2f} '
              f'{r["bbox_diag_pct"]:7.2f} {r["max_radial_pct"]:8.2f} '
              f'{r["centroid_shift"]:7.3f} {r["chamfer_um"]:8.3f} {r["identity_err"]:9.2e}')

    print('\nworst-case error across %d neurons:' % len(rows))
    for k, v in worst.items():
        print(f'  {k:14} {v:6.2f}%   {"OK" if v <= args.tol else "FAIL"}')
    max_ident = max(r['identity_err'] for r in rows)
    print(f'  {"normalise/un-normalise identity":34} max abs err {max_ident:.2e}')
    med_ch = float(np.median([r['chamfer_um'] for r in rows]))
    print(f'  {"chamfer(raw, restored)":34} median {med_ch:.3f} um')

    bad = [k for k, v in worst.items() if v > args.tol]
    if bad:
        sys.exit(f'\nFAIL: {bad} exceed the {args.tol}%% tolerance')
    print(f'\nPASS: all extent statistics within {args.tol}%%')


if __name__ == '__main__':
    main()
