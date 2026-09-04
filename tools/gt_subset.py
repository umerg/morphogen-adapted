#!/usr/bin/env python3
"""Materialise the GT directory matching a generated subset.

`--max_samples N` generates a fixed prefix of the split, so a run holds N neurons out
of a larger val set. Scoring those against the full GT directory compares unequal N,
and MMD / coverage / density all carry N-dependent finite-sample bias --
`run_dist_metrics_cli.py` warns about it but cannot fix it. This builds a GT
directory containing exactly the run's neurons so the comparison is N-matched.

    python tools/gt_subset.py \
        --manifest /scratch/guptau/mg_gen/armS690_c0_bs112/clouds/clouds.json \
        --gt-root  /itet-stor/guptau/net_scratch/neurons_conditional_full/val \
        --out-dir  /scratch/guptau/gt_armS150

Needed twice over: once for a capped smoke-test run, and again for every candidate
in an N-capped checkpoint-selection sweep, where the subset must be identical across
candidates for the comparison to mean anything.

WHAT THIS DOES NOT DO: it does not make the resulting numbers a distributional
measurement. Scoring 150 generated neurons against a 150-neuron reference measures
plumbing, not distributions -- the reference is a sample of the same size drawn from
the same prefix. Use it for the end-to-end gate and for selection where only
*relative* ordering across candidates matters; report headline numbers at the full
split.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


def stems_from_manifest(path: Path) -> list[str]:
    """Source stems named by a v1 clouds.json or a v2 manifest.json.

    Both wrap their rows in a `neurons` list and carry the split-prefixed source id
    as `mid` (e.g. "val/864691...") -- the join key the whole pipeline uses instead
    of position, because the dataloader's `random.Random(38383)` shuffle means
    generated sample i is not split file i.
    """
    doc = json.loads(path.read_text())
    if isinstance(doc, list):
        raise SystemExit(
            f"{path} is a bare-list v1 manifest with no envelope. Regenerate with a "
            "current morphology_gen.py."
        )
    rows = doc.get("neurons")
    if not rows:
        raise SystemExit(f"{path} has no 'neurons' rows")
    stems, seen = [], set()
    for r in rows:
        stem = str(r["mid"]).split("/")[-1]
        if stem in seen:
            raise SystemExit(f"{path} names {stem!r} twice; refusing to build an "
                             "ambiguous reference set")
        seen.add(stem)
        stems.append(stem)
    return stems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True,
                    help="clouds/clouds.json or manifest.json from the generation run")
    ap.add_argument("--gt-root", type=Path, required=True,
                    help="full GT SWC directory for the split, e.g. "
                         "<corpus>/neurons_conditional_full/val")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--copy", action="store_true",
                    help="copy instead of symlinking; use when the target must "
                         "outlive node-local scratch")
    ap.add_argument("--force", action="store_true",
                    help="allow writing into a non-empty --out-dir")
    args = ap.parse_args()

    stems = stems_from_manifest(args.manifest)

    missing = [s for s in stems if not (args.gt_root / f"{s}.swc").is_file()]
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(stems)} manifest neurons are not in "
            f"{args.gt_root}:\n  {missing[:5]}\n"
            "Wrong split, or a corpus that drops neurons this bake kept."
        )

    # A stale .swc left behind would silently join the reference set -- gt_stems()
    # globs the directory, so the adapter would compare against a set nobody chose.
    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.force:
        raise SystemExit(
            f"{args.out_dir} is not empty. Any .swc already there would silently "
            "become part of the reference set. Remove it, pick a fresh path, or "
            "pass --force."
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for s in stems:
        src, dst = (args.gt_root / f"{s}.swc").resolve(), args.out_dir / f"{s}.swc"
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        shutil.copy2(src, dst) if args.copy else os.symlink(src, dst)

    n = len(list(args.out_dir.glob("*.swc")))
    if n != len(stems):
        raise SystemExit(f"wrote {len(stems)} but {n} .swc are present -- "
                         "--out-dir had other files in it")
    print(f"{n} GT neurons -> {args.out_dir}  "
          f"({'copies' if args.copy else 'symlinks'} of {args.gt_root})")


if __name__ == "__main__":
    main()
