#!/usr/bin/env python3
"""Shared conventions for turning MorphoGen reconstruction output into SWC/graphs.

Everything here was previously inline in `tools/recon_ref.py`, with the scale
restore additionally re-implemented in `tools/roundtrip_check.py`. It is
collected in one place so the reconstruction pipeline, the RECON-REF tool and
the dendrite_gen adapter all agree on the conventions -- and, in particular, so
the scale restore cannot drift out of position.

The ordering contract
---------------------
MorphoGen reconstructs in UNIT-SPHERE space. `pc_normlize` divides each baked
neuron by its max radial norm, so the corpus satisfies max||p|| == 1 exactly,
and `detect_radius` / `length_threshold` were calibrated against that (see the
Stage-3 sweeps). The restore must therefore come AFTER the whole reconstruction:

    L1_medial -> fps -> neuron_swc_generator -> filter_short_branches -> auxi
      -> restore_scale_and_clean(...)          <- scale re-enters here

Restoring earlier silently reinterprets tau: at a typical scale_m of ~180 um,
`length_threshold=0.30` would prune at 0.30 um instead of ~54 um, i.e. nothing.

What is NOT a constraint: `clean_swc_tree` itself is scale-equivariant, because
its only length scale is `eps = max(1e-6, eps_scale * local_scale)` with
`local_scale` the mean child distance (dendrite_gen
`preprocessing/clean_trees.py:252`). Verified at k in {1, 177.1, 1e4}: identical
topology, coordinates matching k* to ~1e-16. So cleaning before or after the
restore gives the same answer; the two are bundled here for the contract above,
not because their relative order matters.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

SWC_COLS = ['id', 'type', 'x', 'y', 'z', 'radius', 'parent']

# The exact call the training corpus went through
# (dendrite_gen preprocessing/prepare_conditional_dataset.py:96), so generated
# trees are simplified identically to the GT they are scored against.
CLEAN_KW = dict(root_parent_value=0, keep_parent_value=0,
                max_depth=None, keep_attrs=True, root_mode='index')

MORPHOGEN = Path(__file__).resolve().parent.parent


def _clean_swc_tree():
    """Import dendrite_gen's cleaner lazily.

    It lives in a separate checkout, so importing it at module scope would break
    `python morphology_gen.py` on any machine without dendrite_gen present --
    and generation itself never cleans. Override the location with DENDRITE_GEN.
    """
    dg = Path(os.environ.get('DENDRITE_GEN', Path.home() / 'Documents' / 'dendrite_gen'))
    if str(dg) not in sys.path:
        sys.path.insert(0, str(dg))
    from preprocessing.clean_trees import clean_swc_tree
    return clean_swc_tree


def restore_scale(xyz, scale_m, centroid):
    """Undo `pc_normlize`: unit-sphere coordinates -> the source neuron's um.

    Accepts anything array-like of shape (N, 3). The inverse is affine, not a
    pure scale, so the centroid is required alongside `scale_m`.
    """
    return np.asarray(xyz, dtype=np.float64) * float(scale_m) + np.asarray(centroid, dtype=np.float64)


def restore_scale_and_clean(nodes, scale_m, centroid, clean_kw=None):
    """auxi output (unit-sphere) -> um -> `clean_swc_tree` -> cleaned DataFrame.

    `nodes` is either the list of 7-tuples `auxi` returns or a DataFrame with
    SWC_COLS. Callers must already have run `filter_short_branches` and `auxi`
    on the unit-sphere tree -- see the ordering contract in the module docstring.

    Why there is no `to_one_indexed` call here, which looks like an omission and is
    not: `neuron_swc_generator` emits 0-INDEXED output (soma id 0, parent -1, so its
    children carry parent == 0), which `clean_swc_tree` would shatter into a forest
    exactly as `to_one_indexed` warns. But `auxi` renumbers on the way through --
    verified by running the real chain: ids 0..240 with parents {-1, 0, 1, ...} in,
    ids 1..311 with parents {-1, 1, 2, ...} out. So by the time anything reaches this
    function the root is the only node with parent <= 0, which is the invariant
    `clean_swc_tree(root_parent_value=0)` needs.

    The consequence for callers: this function is safe for `auxi` output ONLY. Feed
    it a raw 0-indexed SWC that has not been through `auxi` -- a corpus file, say --
    and it will silently return a forest. Use `to_one_indexed` on that path instead,
    as `recon_ref.py` does for the GT side.
    """
    import pandas as pd

    df = nodes if hasattr(nodes, 'columns') else pd.DataFrame(nodes, columns=SWC_COLS)
    df = df.copy()
    df[['x', 'y', 'z']] = restore_scale(df[['x', 'y', 'z']].to_numpy(), scale_m, centroid)
    return _clean_swc_tree()(df, **(CLEAN_KW if clean_kw is None else clean_kw))


def to_one_indexed(df):
    """Shift a 0-indexed SWC to the 1-indexed convention dendrite_gen expects.

    CRITICAL: the raw MICrONS corpus is 0-indexed -- the soma is id 0 with parent -1,
    so its children carry parent == 0. But clean_swc_tree treats `parent <= root_parent_value`
    (=0) as "this node IS a root". Fed 0-indexed input it therefore promotes every primary
    dendrite to its own root and returns a FOREST, silently shattering each neuron into
    ~7 disconnected components rather than raising. Shift ids by +1 so parent==0 again
    means "root" and nothing else.
    """
    if len(df) == 0 or int(df['id'].min()) != 0:
        return df
    out = df.copy()
    out['id'] = out['id'].astype(int) + 1
    out['parent'] = np.where(df['parent'].astype(int) < 0, 0, df['parent'].astype(int) + 1)
    return out


def df_to_graph(df):
    """clean_swc_tree output -> rooted nx tree, matching utils.data_loading.load_swc_graph."""
    import networkx as nx
    G = nx.Graph()
    pos = {}
    root = None
    for r in df.itertuples(index=False):
        G.add_node(int(r.id))
        pos[int(r.id)] = np.array([r.x, r.y, r.z], float)
        if int(r.parent) <= 0:
            root = int(r.id)
    for r in df.itertuples(index=False):
        if int(r.parent) > 0:
            G.add_edge(int(r.parent), int(r.id))
    if root is None:
        raise ValueError('no root: no node has parent <= 0')
    if nx.number_connected_components(G) != 1:
        raise ValueError(f'expected a tree, got {nx.number_connected_components(G)} components '
                         '(0-indexed input fed to a 1-indexed consumer?)')
    origin = pos[root]
    for n in G.nodes:
        G.nodes[n]['pos'] = pos[n] - origin
    G.graph['root'] = root
    return G
