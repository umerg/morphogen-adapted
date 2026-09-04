"""Score generated trees against a GT split — the headline distributional table.

Lives in MorphoGen so the baseline owns its pipeline end to end:

    --stage sample -> --stage reconstruct -> --stage finalize -> tools/score_trees.py

The metric itself belongs to dendrite_gen (`validation/dist_metrics.py`); this is a
runner over it, importing that repo lazily via DENDRITE_GEN. Nothing is added to
dendrite_gen.

`compute_distribution_metrics` had no CLI: it was reachable only from
`graph_generation/training.py` and two `data_analysis/` scripts, so the headline
`mmd_morpho` / `density_morpho` / `coverage_morpho` numbers could not be produced
for anything trained outside this repo. That blocks every baseline comparison, and
it blocks checkpoint selection for baselines whose trainer lives elsewhere.

This mirrors the in-training call exactly — `training.py:891-908` for the metric,
`:594-624` for the real-vs-real floor — so a number produced here is the same
number the trainer would log.

Two things are easy to get wrong and are handled explicitly:

* **`uhat` must be the y-axis for neurons.** `compute_tmd_embedding` defaults to z.
  Harmless for the default `radial_root` filtration, which is axis-agnostic, but
  that makes a wrong axis invisible rather than absent. Pass `--so2-axis 0 1 0`.
* **The floor must reuse the GT cache fitted on the eval set**, and must draw its
  train subset with a fixed seed sized to `len(eval)`, or the excess is not
  comparable across runs.

Usage
-----
  python tools/score_trees.py \\
      --gt-dir <val SWC dir> --pred-dir <MorphoGen .../trees> \\
      --train-dir <train SWC dir> --so2-axis 0 1 0 \\
      --output-json morphogen_val_dist.json
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import networkx as nx
import numpy as np

def _dendrite_gen():
    """Put dendrite_gen on the path, lazily, and FIRST.

    The metric belongs to dendrite_gen (`validation/dist_metrics.py`); this file is a
    runner over it, kept here so the MorphoGen baseline owns its whole pipeline and
    dendrite_gen carries none of it. Same bootstrap direction as
    `tools/morphogen_swc.py`, which reaches across for `clean_swc_tree`.

    Position matters: BOTH repos have a top-level `utils` package, and MorphoGen's has
    no `data_loading`. Inserting dendrite_gen at sys.path[0] -- and never adding the
    MorphoGen root here, which the previous in-repo version did -- is what makes
    `utils.data_loading` resolve. Do not import MorphoGen modules from this file.
    """
    dg = Path(os.environ.get("DENDRITE_GEN", Path.home() / "Documents" / "dendrite_gen"))
    if not (dg / "validation" / "dist_metrics.py").is_file():
        raise SystemExit(
            f"no dendrite_gen checkout at {dg} (looked for validation/dist_metrics.py).\n"
            "  Set DENDRITE_GEN=/path/to/dendrite_gen."
        )
    sys.path.insert(0, str(dg))


_dendrite_gen()

from utils.data_loading import load_swc_graph, load_swc_graphs_from_dir  # noqa: E402
from utils.tmd import compute_tmd_embedding  # noqa: E402
from validation.dist_metrics import build_gt_cache, compute_distribution_metrics  # noqa: E402


def load_pred_graphs(pred_pkl: Path, ema_key: str | None) -> list[nx.Graph]:
    """Accept every payload shape `build_output_payload` can emit."""
    with pred_pkl.open("rb") as fh:
        payload = pickle.load(fh)

    if isinstance(payload, list):
        return payload
    if "pred_graphs" in payload:
        return payload["pred_graphs"]

    keys = [k for k in payload if isinstance(payload[k], dict) and "pred_graphs" in payload[k]]
    if not keys:
        raise ValueError(f"{pred_pkl}: no pred_graphs found; top-level keys = {list(payload)}")
    if ema_key is None:
        if len(keys) > 1:
            raise ValueError(
                f"{pred_pkl} holds several EMA keys {keys}; pass --ema-key to choose."
            )
        ema_key = keys[0]
    if ema_key not in payload:
        raise KeyError(f"{pred_pkl}: no key {ema_key!r}; available: {keys}")
    return payload[ema_key]["pred_graphs"]


def repair_roots(graphs: list[nx.Graph], label: str) -> int:
    """Mirror training.py:874-877 — a graph with no usable root gets node 0.

    Counted and reported rather than silently applied: on a well-formed adapter
    output this must be 0, so a non-zero count is a bug upstream, not a nuisance.
    """
    fixed = 0
    for G in graphs:
        if "root" not in G.graph or G.graph.get("root") not in G.nodes:
            G.graph["root"] = next(iter(G.nodes)) if G.number_of_nodes() else 0
            fixed += 1
    if fixed:
        print(f"  WARNING: repaired {fixed}/{len(graphs)} {label} graphs with no usable root")
    return fixed


def main() -> None:
    p = argparse.ArgumentParser(description="Run the distribution-metric suite from the CLI.")
    p.add_argument("--gt-dir", type=Path, required=True,
                   help="Evaluation split SWC directory (the reference distribution).")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--pred-dir", type=Path, default=None,
                     help="Directory of predicted SWCs -- e.g. MorphoGen's "
                          "<generate_dir>/trees from --stage finalize. Loaded with "
                          "the SAME loader as --gt-dir, so both sides go through one "
                          "code path. Preferred over --pred-pkl for anything that "
                          "emits SWC natively.")
    src.add_argument("--pred-pkl", type=Path, default=None,
                     help="Prediction pickle, for methods that do not emit SWC "
                          "(e.g. SemlaFlow via convert_smol_to_pred_pkl.py).")
    p.add_argument("--ema-key", default=None,
                   help="Key inside the pickle; inferred when there is exactly one.")
    p.add_argument("--train-dir", type=Path, default=None,
                   help="Train SWC directory for the real-vs-real floor. Without it, "
                        "headline_excess_mmd_morpho cannot be computed.")
    p.add_argument("--so2-axis", type=float, nargs=3, default=[0.0, 0.0, 1.0],
                   help="SO(2) symmetry axis. USE 0 1 0 FOR NEURONS.")
    p.add_argument("--tmd-filtration", default="radial_root")
    p.add_argument("--tmd-bins", type=int, default=16)
    p.add_argument("--tmd-pca-ncomp", type=int, default=32)
    p.add_argument("--dc-k", type=int, default=5)
    p.add_argument("--morpho-whiten", action="store_true")
    p.add_argument("--ged", action="store_true",
                   help="Enable tree edit distance. Off by default: it is the slowest term "
                        "and training disables it for the floor regardless.")
    p.add_argument("--floor-seed", type=int, default=0,
                   help="Seed for the train subsample; 0 matches training.py:601.")
    p.add_argument("--output-json", type=Path, default=None)
    args = p.parse_args()

    uhat = tuple(np.asarray(args.so2_axis, dtype=float).reshape(3).tolist())
    if uhat == (0.0, 0.0, 1.0):
        print("NOTE: --so2-axis is the z default. Neurons use the y axis (0 1 0).")

    gt_graphs = load_swc_graphs_from_dir(args.gt_dir)
    if args.pred_dir is not None:
        # Same loader as the GT side, and it sorts by filename -- so gen and GT are
        # index-aligned by construction. The pred-pkl path had to reorder explicitly
        # to achieve that; here it is free.
        pred_graphs = load_swc_graphs_from_dir(args.pred_dir)
    else:
        pred_graphs = load_pred_graphs(args.pred_pkl, args.ema_key)
    print(f"GT graphs   : {len(gt_graphs)} from {args.gt_dir}")
    print(f"pred graphs : {len(pred_graphs)} from {args.pred_dir or args.pred_pkl}")
    if len(pred_graphs) != len(gt_graphs):
        # Not fatal — the metrics are distributional — but N-dependent finite-sample
        # bias makes an unmatched comparison misleading, so say so loudly.
        print(f"  WARNING: N differs ({len(pred_graphs)} vs {len(gt_graphs)}). "
              "MMD/coverage/density carry N-dependent bias; matched N is required "
              "for a headline table.")
    repair_roots(gt_graphs, "GT")
    repair_roots(pred_graphs, "pred")

    embed_fn = (lambda G: compute_tmd_embedding(
        G, filtration=args.tmd_filtration, n_bins=args.tmd_bins, uhat=uhat))
    shared = dict(tmd_filtration=args.tmd_filtration, morpho_whiten=args.morpho_whiten)

    gt_cache = build_gt_cache(gt_graphs, uhat=uhat, embed_fn=embed_fn,
                              tmd_pca_ncomp=args.tmd_pca_ncomp,
                              morpho_whiten=args.morpho_whiten)

    print("computing distribution metrics ...")
    dist = compute_distribution_metrics(
        pred_graphs, gt_graphs, uhat=uhat, ged_enabled=args.ged, gt_cache=gt_cache,
        embed_fn=embed_fn, dc_k=args.dc_k, tmd_pca_ncomp=args.tmd_pca_ncomp, **shared)

    out = {"config": {"gt_dir": str(args.gt_dir),
                      "pred_source": str(args.pred_dir or args.pred_pkl),
                      "ema_key": args.ema_key, "so2_axis": list(uhat),
                      "n_gt": len(gt_graphs), "n_pred": len(pred_graphs),
                      "tmd_filtration": args.tmd_filtration, "ged": args.ged},
           "dist": dist}

    if args.train_dir:
        # Real-vs-real floor, exactly as training.py:594-624: an N-matched train
        # subset scored against the SAME eval set, reusing the SAME gt_cache, with a
        # fixed seed. GED off, as it is there.
        train_graphs = load_swc_graphs_from_dir(args.train_dir)
        if not train_graphs:
            # Silently returning a nan floor would make headline_excess vanish from
            # the output with no explanation, which reads as "not computed" rather
            # than "you pointed me at an empty directory".
            raise SystemExit(f"--train-dir {args.train_dir} contains no .swc files; "
                             "the real-vs-real floor cannot be computed.")
        repair_roots(train_graphs, "train")
        rng = np.random.default_rng(args.floor_seed)
        n = len(gt_graphs)
        if len(train_graphs) > n:
            idx = rng.choice(len(train_graphs), size=n, replace=False)
            train_sub = [train_graphs[i] for i in idx]
        else:
            train_sub = list(train_graphs)
        print(f"computing real-vs-real floor ({len(train_sub)} train vs {n} eval) ...")
        floor = compute_distribution_metrics(
            train_sub, gt_graphs, uhat=uhat, ged_enabled=False, gt_cache=gt_cache,
            embed_fn=embed_fn, dc_k=args.dc_k, tmd_pca_ncomp=args.tmd_pca_ncomp, **shared)
        out["floor"] = floor
        gen_mmd = dist.get("mmd_morpho", float("nan"))
        floor_mmd = floor.get("mmd_morpho", float("nan"))
        if np.isfinite(gen_mmd) and np.isfinite(floor_mmd):
            out["dist"]["headline_excess_mmd_morpho"] = float(gen_mmd - floor_mmd)

    print("\n=== headline ===")
    for k in ("mmd_morpho", "density_morpho", "coverage_morpho",
              "headline_excess_mmd_morpho", "morpho_gt_nan_frac"):
        if k in out["dist"]:
            print(f"  {k:32s} {out['dist'][k]:.6f}")
    if "floor" in out and "mmd_morpho" in out["floor"]:
        print(f"  {'floor mmd_morpho':32s} {out['floor']['mmd_morpho']:.6f}")

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(out, indent=2, default=float))
        print(f"\nwrote {args.output_json}")


if __name__ == "__main__":
    main()
