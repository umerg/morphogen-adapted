#!/usr/bin/env python3
"""Build a single-class subset synset from an existing bake -- no re-baking.

Arm S (plan Stage 6b) trains on ~690 neurons of one cell class at Arm P's exact
gradient-step budget, so per-neuron exposure rises ~34x and nothing else about the
budget changes. That needs a corpus directory holding only those neurons.

Nothing has to be re-baked to get one. This reads the `.meta.json` sidecars beside
the baked `.npy` files, picks a class, and copies the selected neurons into a fresh
synset tree.

**The 2026-08 bake carries `cell_class: null` everywhere** (plan section 11 Stage 5)
even though `scripts/bake.sbatch:133` passes `--label-root` -- the labels never
reached the sidecars. They are not lost: they live in the labelled SWC corpus and
join on filename, so `--label-root` recovers them here. The point clouds are
untouched by this; it is purely an annotation that went missing.

Two deliberate choices:

* **Copy, not symlink.** The corpus usually lives on node-local `/scratch`, and a
  symlink into it dangles the moment a job lands on a different node. 690 neurons
  is ~124 MB, so copying costs nothing and the result is portable.
* **The synset is named `neurons` by default.** `scripts/train_armU.sbatch`
  hardcodes `--category neurons` and exposes everything else through the
  environment, so keeping the name means Arm S runs that script unmodified.

Usage
-----
  # 1. what classes are there, and how many of each?
  python tools/make_subset_synset.py --npy-root <bake> \\
      --label-root <share>/neurons_conditional_full --counts

  # 2. build the subset
  python tools/make_subset_synset.py --npy-root <bake> \\
      --label-root <share>/neurons_conditional_full \\
      --out-root <bake>_c0_690 --cell-class 0 --n 690
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.swc_to_morphogen_npy import read_cell_class  # noqa: E402

SPLITS = ('train', 'val')
NPOINTS = 15000


def meta_path(npy: Path) -> Path:
    """`foo.npy` -> `foo.meta.json`.

    Sliced rather than `Path.with_suffix`, which would eat the last dot-bearing
    component of an id like `864691135082733559_393360.v2`.
    """
    return npy.parent / (npy.name[:-len('.npy')] + '.meta.json')


def scan(npy_root: Path, synset: str,
         label_root: Path | None = None) -> dict[str, list[tuple[Path, int | None]]]:
    """Every baked neuron per split, with its cell_class.

    Read from the sidecar when it is there. The 2026-08 bake ran with a
    --label-root the baker could not use, so every sidecar carries
    `cell_class: null` (plan section 11 Stage 5) -- and the labels are NOT lost, they
    live in the labelled SWC corpus and join on filename. Passing --label-root
    recovers them without re-baking a single point cloud: the .npy files are fine,
    it is only the annotation that is missing.
    """
    out: dict[str, list[tuple[Path, int | None]]] = {}
    n_recovered = 0
    for split in SPLITS:
        d = npy_root / synset / split
        if not d.is_dir():
            raise SystemExit(f'no such split directory: {d}')
        rows = []
        for npy in sorted(d.glob('*.npy')):
            mp = meta_path(npy)
            if not mp.is_file():
                raise SystemExit(
                    f'{npy} has no sidecar at {mp}. The bake is incomplete; a subset '
                    'cannot be built without scale_m/centroid for every neuron.')
            meta = json.loads(mp.read_text())
            cls = meta.get('cell_class')
            if cls is None and label_root is not None:
                stem = npy.name[:-len('.npy')]
                cls = read_cell_class(label_root / split / f'{stem}.swc')
                if cls is not None:
                    n_recovered += 1
            rows.append((npy, cls))
        if not rows:
            raise SystemExit(f'{d} contains no .npy files')
        out[split] = rows
    if n_recovered:
        print(f'recovered {n_recovered} labels from {label_root} by filename join')
    return out


def report_counts(scanned) -> None:
    for split, rows in scanned.items():
        c = Counter(cls for _, cls in rows)
        total = len(rows)
        print(f'{split}: {total} neurons')
        for cls in sorted(c, key=lambda k: (k is None, k)):
            label = 'NULL (unlabelled)' if cls is None else f'class {cls}'
            print(f'    {label:<20} {c[cls]:6d}  ({100.0 * c[cls] / total:5.1f}%)')


def require_labels(scanned) -> None:
    """A bake made without --label-root has cell_class=null everywhere.

    Fail here rather than silently building an empty or arbitrary subset: the fix
    is to re-run the bake with --label-root, which is a different job entirely.
    """
    for split, rows in scanned.items():
        n_null = sum(1 for _, cls in rows if cls is None)
        if n_null == len(rows):
            raise SystemExit(
                f'every sidecar under {split} has cell_class=null, and no labels were '
                'recovered.\n'
                '  The .npy clouds are fine -- only the annotation is missing, and it '
                'joins by filename.\n'
                '  Pass --label-root <labelled SWC corpus> (e.g. '
                '.../neurons_conditional_full);\n'
                '  re-baking is NOT needed. If --label-root was already passed, the '
                'filename join\n'
                '  found nothing: check that <label-root>/<split>/<stem>.swc exists and '
                'carries a\n  leading "# cell_class N" comment.')
        if n_null:
            raise SystemExit(
                f'{n_null}/{len(rows)} sidecars under {split} have cell_class=null. '
                'A partial labelling would bias the subset in a way that is invisible '
                'downstream; fix the bake first.')


def copy_neuron(npy: Path, dst_dir: Path, cell_class: int | None = None) -> None:
    shutil.copy2(npy, dst_dir / npy.name)
    mp = meta_path(npy)
    dst_meta = dst_dir / mp.name
    shutil.copy2(mp, dst_meta)
    if cell_class is not None:
        # Write the resolved label into the COPY, never the source bake. Without this
        # the subset would still read as unlabelled, and Arm C later needs the label
        # to reach `cate_idx`.
        meta = json.loads(dst_meta.read_text())
        if meta.get('cell_class') is None:
            meta['cell_class'] = cell_class
            meta['cell_class_source'] = 'filename join (--label-root)'
            dst_meta.write_text(json.dumps(meta))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--npy-root', type=Path, required=True,
                    help='the bake root, i.e. the directory CONTAINING the synset dir')
    ap.add_argument('--synset', default='neurons')
    ap.add_argument('--out-root', type=Path, default=None,
                    help='where to write <out-root>/<out-synset>/{train,val}')
    ap.add_argument('--out-synset', default=None,
                    help='name of the synset written out. Defaults to --synset, which '
                         'is what lets scripts/train_armU.sbatch run unmodified.')
    ap.add_argument('--cell-class', type=int, default=None)
    ap.add_argument('--n', type=int, default=690,
                    help='train neurons to select (0 = all of that class). 690 matches '
                         'the paper: ~690-760 train neurons per model.')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--label-root', type=Path, default=None,
                    help='labelled SWC corpus (e.g. .../neurons_conditional_full). '
                         'Use when the bake sidecars carry cell_class=null: labels are '
                         'joined by filename, so nothing has to be re-baked.')
    ap.add_argument('--counts', action='store_true',
                    help='report the class histogram and exit, changing nothing')
    args = ap.parse_args()

    scanned = scan(args.npy_root, args.synset, args.label_root)
    report_counts(scanned)
    if args.counts:
        return
    if args.cell_class is None:
        raise SystemExit('\npass --cell-class K to build a subset (or --counts to stop here)')
    if args.out_root is None:
        raise SystemExit('--out-root is required when building a subset')
    require_labels(scanned)

    pools = defaultdict(list)
    for split, rows in scanned.items():
        pools[split] = [npy for npy, cls in rows if cls == args.cell_class]
    if not pools['train']:
        raise SystemExit(f'no train neurons with cell_class={args.cell_class}')
    if len(pools['train']) < args.n:
        raise SystemExit(
            f'asked for {args.n} train neurons of class {args.cell_class} but only '
            f'{len(pools["train"])} exist. Pick a more populous class or lower --n.')

    # Seeded selection over a filename-sorted pool, so the same flags reproduce the
    # same subset on any machine -- `glob` order is filesystem-dependent, `sorted`
    # is not.
    rng = np.random.default_rng(args.seed)
    train_pool = sorted(pools['train'])
    if args.n:
        idx = rng.choice(len(train_pool), size=args.n, replace=False)
        train_sel = [train_pool[i] for i in sorted(idx)]
    else:
        train_sel = train_pool
    # ALL val neurons of the class: generation caps its own sample count with
    # --max_samples, so there is no reason to throw held-out data away here.
    val_sel = sorted(pools['val'])

    out_synset = args.out_synset or args.synset
    print(f'\nselecting class {args.cell_class}: '
          f'{len(train_sel)} train (of {len(train_pool)}), {len(val_sel)} val')

    selected = {'train': train_sel, 'val': val_sel}
    for split, files in selected.items():
        dst = args.out_root / out_synset / split
        dst.mkdir(parents=True, exist_ok=True)
        for i, npy in enumerate(files, 1):
            copy_neuron(npy, dst, args.cell_class)
            if i % 100 == 0 or i == len(files):
                print(f'  {split}: {i}/{len(files)}', flush=True)

    manifest = {
        'source_npy_root': str(args.npy_root),
        'source_synset': args.synset,
        'out_synset': out_synset,
        'cell_class': args.cell_class,
        'n_requested': args.n,
        'seed': args.seed,
        'counts': {k: len(v) for k, v in selected.items()},
        'files': {k: [p.name for p in v] for k, v in selected.items()},
    }
    (args.out_root / 'subset.json').write_text(json.dumps(manifest, indent=1))

    # Verify what we wrote rather than trusting the copy: a short .npy from a full
    # disk would otherwise surface as an assert inside the dataloader hours later.
    bad = 0
    for split, files in selected.items():
        dst = args.out_root / out_synset / split
        for npy in files:
            arr = np.load(dst / npy.name, mmap_mode='r')
            if arr.shape != (NPOINTS, 3):
                print(f'  BAD SHAPE {npy.name}: {arr.shape}', file=sys.stderr)
                bad += 1
    if bad:
        raise SystemExit(f'{bad} copied .npy files have the wrong shape')

    print(f'\nwrote {args.out_root}/{out_synset}/{{train,val}} + subset.json')
    print(f'  train {len(train_sel)}  val {len(val_sel)}  '
          f'(all verified ({NPOINTS}, 3))')
    print(f'\nsteps/epoch at bs 112 (drop_last): {len(train_sel) // 112}')


if __name__ == '__main__':
    main()
