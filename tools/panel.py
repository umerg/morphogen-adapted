#!/usr/bin/env python3
"""One PNG: generated clouds beside the trees reconstructed from them.

The Arm S readout is a number (local linearity against `gt_baseline.json`), but the
number is a summary of something you can see directly: a filamentary cloud looks
like a neuron, a diffuse one looks like a smudge shaped like a neuron. This renders
both, plus optionally the paired ground truth, so the metric and the picture can be
checked against each other.

    python tools/panel.py --generate-dir $GEN --n 8 --out panel.png
    python tools/panel.py --generate-dir $GEN --n 8 --out panel.png \
        --ref-npy-root $SUBSET/neurons --ref-split val \
        --gt-swc-dir /itet-stor/guptau/net_scratch/neurons_conditional_full/val

Columns are whatever is available: GT cloud | generated cloud | generated tree |
GT tree. Trees appear once `--stage reconstruct` has run; without it the tree
columns say so and print the command.

EVERY PANEL IS NORMALISED INDEPENDENTLY -- centred on its centroid and divided by
its max radial norm -- so the figure compares SHAPE only. Absolute scale is
deliberately not readable here: generated SWCs are unit-sphere until the adapter
restores `gt_scale_m`, and `gen_radius` is printed per row instead, which is the
honest place to read scale off.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use('Agg')          # cluster nodes have no display
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.cloud_quality import linearity, gen_radius, unit_sphere  # noqa: E402


# --------------------------------------------------------------------------- io

def read_swc(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """-> (xyz [N,3], parent_row [N]), parent_row = -1 at the root.

    Handles both conventions in play: the generated SWCs are written by
    `morphology_gen.reconstruct_clouds` as bare `id type x y z radius parent` with
    the soma at parent -1, while the dendrite_gen corpus is 1-indexed with
    `# cell_class` header comments and root_parent_value 0. Any `parent <= 0` is a
    root, matching dendrite_gen's `load_swc_graph`.
    """
    ids, xyz, par = [], [], []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        f = line.split()
        if len(f) < 7:
            continue
        ids.append(int(float(f[0])))
        xyz.append([float(f[2]), float(f[3]), float(f[4])])
        par.append(int(float(f[6])))
    if not ids:
        raise ValueError(f'{path} has no SWC rows')
    row_of = {i: r for r, i in enumerate(ids)}
    # Which parent values mean "root" depends on the indexing convention, and both
    # are in play. The raw MICrONS corpus is 0-INDEXED with the soma at id 0 and
    # parent -1, so `parent == 0` is a real soma edge; dendrite_gen's cleaned corpus
    # is 1-indexed with root_parent_value 0, so `parent == 0` is the root. Reading a
    # 0-indexed file with the 1-indexed rule silently orphans every soma child --
    # the same off-by-one class as the `getBranch` `int(id)-1` bug.
    zero_indexed = min(ids) == 0
    def _root(p):
        return p < 0 if zero_indexed else p <= 0
    parent_row = np.array([-1 if _root(p) else row_of.get(p, -1) for p in par])
    return np.asarray(xyz, dtype=np.float64), parent_row


def ref_cloud(path: Path, npoints: int, rng: np.random.Generator) -> np.ndarray:
    """A GT cloud built the way the dataloader builds it.

    Mirrors `cloud_quality.load_reference` (:120): `Uniform15KPC` keeps
    `all_points[:, 10000:]` and draws a random subset of that. The 15,000-point bake
    is FPS-ordered, so the last 5,000 are density infill with different local
    statistics -- taking a uniform draw over all 15,000 would make the GT linearity
    incomparable to the generated one.
    """
    cloud = np.load(path).astype(np.float64)
    tail = cloud[10000:]
    return tail[rng.choice(len(tail), min(npoints, len(tail)), replace=False)]


# ---------------------------------------------------------------- tree measures

def tree_stats(xyz: np.ndarray, parent_row: np.ndarray) -> dict:
    n = len(xyz)
    nkids = np.zeros(n, dtype=int)
    for p in parent_row:
        if p >= 0:
            nkids[p] += 1
    roots = np.flatnonzero(parent_row < 0)
    order = np.zeros(n, dtype=int)
    # Adjacency once, not a scan of parent_row per node: a dense reconstruction is
    # a few thousand nodes and the naive form is quadratic in it.
    kids = [[] for _ in range(n)]
    for c, p in enumerate(parent_row):
        if p >= 0:
            kids[p].append(c)
    # Children usually follow their parent in an SWC, but do not rely on it.
    seen, stack = set(roots.tolist()), list(roots.tolist())
    while stack:
        v = stack.pop()
        for c in kids[v]:
            if c not in seen:
                order[c] = order[v] + (1 if nkids[v] > 1 else 0)
                seen.add(c)
                stack.append(c)
    return {'nodes': n,
            'leaves': int((nkids == 0).sum()),
            'root_deg': int(nkids[roots[0]]) if len(roots) else 0,
            'max_order': int(order.max())}


# ------------------------------------------------------------------- rendering

def orient(P: np.ndarray, uhat_axis: int) -> np.ndarray:
    """Unit-sphere normalise, then put the corpus axis on the plot's vertical.

    Neurons use uhat = y (`config/parity_neurons_uncond.yaml:117`) but matplotlib
    draws z upward, so without the permutation every neuron renders on its side and
    apical-vs-basal -- the thing these plots are for -- is unreadable.
    """
    P = unit_sphere(P)
    rest = [a for a in (0, 1, 2) if a != uhat_axis]
    return P[:, [rest[0], rest[1], uhat_axis]]


def style(ax) -> None:
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.set_axis_off()


def draw_cloud(ax, P, title, elev, azim, color):
    Q = orient(P, ax._uhat)
    ax.scatter(Q[:, 0], Q[:, 1], Q[:, 2], s=0.7, c=color, alpha=0.45,
               linewidths=0, rasterized=True)
    ax.view_init(elev=elev, azim=azim)
    style(ax)
    ax.set_title(title, fontsize=7, pad=1)


def draw_tree(ax, xyz, parent_row, title, elev, azim, color):
    Q = orient(xyz, ax._uhat)
    seg = [[Q[c], Q[p]] for c, p in enumerate(parent_row) if p >= 0]
    ax.add_collection3d(Line3DCollection(seg, colors=color, linewidths=0.6))
    roots = np.flatnonzero(parent_row < 0)
    if len(roots):
        ax.scatter(Q[roots, 0], Q[roots, 1], Q[roots, 2], s=18, c='crimson',
                   depthshade=False, zorder=5)
    ax.view_init(elev=elev, azim=azim)
    style(ax)
    ax.set_title(title, fontsize=7, pad=1)


def draw_missing(ax, msg):
    ax.text2D(0.5, 0.5, msg, ha='center', va='center', fontsize=7,
              color='0.5', transform=ax.transAxes)
    ax.set_axis_off()


# ------------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--generate-dir', type=Path, required=True,
                    help='the --generate_dir from morphology_gen; holds clouds/ '
                         'and, once reconstructed, *.swc')
    ap.add_argument('--clouds-dir', type=Path, default=None,
                    help='override; defaults to <generate-dir>/clouds')
    ap.add_argument('--n', type=int, default=8, help='neurons to show')
    ap.add_argument('--pick', default='', help='comma-separated mids, overrides --n')
    ap.add_argument('--out', type=Path, default=Path('panel.png'))
    ap.add_argument('--dpi', type=int, default=170)
    ap.add_argument('--elev', type=float, default=14.0)
    ap.add_argument('--azim', type=float, default=-62.0)
    ap.add_argument('--uhat-axis', type=int, default=1,
                    help='corpus axis index; 1 = y, the neuron convention')
    ap.add_argument('--ref-npy-root', type=Path, default=None,
                    help='<subset>/neurons -- adds a paired GT cloud column')
    ap.add_argument('--ref-split', default='val')
    ap.add_argument('--gt-swc-dir', type=Path, default=None,
                    help='GT SWC directory -- adds a paired GT tree column')
    ap.add_argument('--npoints', type=int, default=2048)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    cdir = args.clouds_dir or (args.generate_dir / 'clouds')
    idx_path = cdir / 'clouds.json'
    if not idx_path.is_file():
        raise SystemExit(f'no {idx_path}; run --stage sample first.')
    index = json.load(open(idx_path))['neurons']

    if args.pick:
        want = [m.strip() for m in args.pick.split(',') if m.strip()]
        rows = [r for r in index if r['mid'] in want or Path(r['mid']).name in want]
        missing = set(want) - {r['mid'] for r in rows} - {Path(r['mid']).name for r in rows}
        if missing:
            raise SystemExit(f'not in clouds.json: {sorted(missing)}')
    else:
        # Evenly spaced rather than the first n, so the panel is not all drawn from
        # one corner of the fixed prefix.
        k = min(args.n, len(index))
        rows = [index[i] for i in np.linspace(0, len(index) - 1, k).astype(int)]

    have_trees = any((args.generate_dir / r['file']).is_file() for r in rows)
    if not have_trees:
        print('no .swc alongside the clouds -- the tree columns will be empty.\n'
              'Reconstruct them with:\n'
              f'  python -u morphology_gen.py --stage reconstruct \\\n'
              f'      --generate_dir {args.generate_dir} --recon_workers 8 \\\n'
              f'      --detect_radius 0.30 --root_cap 23\n')

    cols = ['gen cloud', 'gen tree']
    if args.ref_npy_root:
        cols.insert(0, 'GT cloud')
    if args.gt_swc_dir:
        cols.append('GT tree')

    rng = np.random.default_rng(args.seed)
    fig = plt.figure(figsize=(2.15 * len(cols), 2.35 * len(rows)))
    for i, row in enumerate(rows):
        stem = Path(row['mid']).name
        for j, col in enumerate(cols):
            ax = fig.add_subplot(len(rows), len(cols), i * len(cols) + j + 1,
                                 projection='3d')
            ax._uhat = args.uhat_axis
            try:
                if col == 'gen cloud':
                    P = np.load(cdir / row['cloud']).astype(np.float64)
                    lin, nn = linearity(P)
                    draw_cloud(ax, P, f'{stem}\nlin {lin:.3f}  nn {nn:.4f}  '
                                      f'r {row["gen_radius"]:.2f}',
                               args.elev, args.azim, '#1f77b4')
                elif col == 'GT cloud':
                    P = ref_cloud(args.ref_npy_root / args.ref_split / f'{stem}.npy',
                                  args.npoints, rng)
                    lin, nn = linearity(P)
                    draw_cloud(ax, P, f'GT {stem}\nlin {lin:.3f}  nn {nn:.4f}',
                               args.elev, args.azim, '#2ca02c')
                elif col == 'gen tree':
                    xyz, par = read_swc(args.generate_dir / row['file'])
                    s = tree_stats(xyz, par)
                    draw_tree(ax, xyz, par,
                              'recon  n {nodes}  lv {leaves}  '
                              'k {root_deg}  ord {max_order}'.format(**s),
                              args.elev, args.azim, '#111111')
                else:
                    xyz, par = read_swc(args.gt_swc_dir / f'{stem}.swc')
                    s = tree_stats(xyz, par)
                    draw_tree(ax, xyz, par,
                              'GT  n {nodes}  lv {leaves}  '
                              'k {root_deg}  ord {max_order}'.format(**s),
                              args.elev, args.azim, '#2ca02c')
            except FileNotFoundError:
                draw_missing(ax, f'{col}\nnot found')
            except Exception as e:                       # noqa: BLE001
                draw_missing(ax, f'{col}\n{type(e).__name__}')
                print(f'{stem} / {col}: {e}', file=sys.stderr)

    fig.suptitle(f'{args.generate_dir.name}   '
                 f'(each panel unit-sphere normalised; scale is in gen_radius)',
                 fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches='tight')
    print(f'wrote {args.out}  ({len(rows)} neurons x {len(cols)} columns)')


if __name__ == '__main__':
    main()
