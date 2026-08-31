#!/usr/bin/env python3
"""Score generated point clouds WITHOUT reconstructing them.

Reconstruction costs ~3 s/neuron of CPU; sampling a checkpoint and looking at its
clouds costs nothing extra once `--stage sample` has run. Everything here is
computed on clouds alone, which makes it usable three ways:

1. **Arm S readout** (plan Stage 6b). The finding that motivated Arm S is a cloud
   property, not a tree property: the epoch-299 Arm U model has the right envelope
   (`gen_radius` 1.08 against a training target of exactly 1.0) but its points do
   not lie on filaments (linearity 0.605 against GT 0.975). Skeletonising a fuzzy
   cloud is what inflates node count 2.66x, and no tau fixes it.
2. **Checkpoint pre-filter** (plan section 6). A checkpoint whose `gen_radius` is far
   from 1.0 is out of distribution, and its morphometrics are an artifact of
   applying unit-sphere constants at the wrong scale -- so it should be rejected
   before anything is reconstructed. Measured: Arm U epoch 99 sits at 63.5.
3. **Comparison against the paper on its own terms.** MorphoGen reports MMD/COV over
   Chamfer on clouds downsampled to 128 points (their section 5.3), which is blind to
   filament structure and to scale. Reporting that number next to linearity is what
   makes the difference legible.

Usage
-----
  # generated clouds vs the val split they should resemble
  python tools/cloud_quality.py --clouds <gen>/clouds \\
      --ref-npy-root <bake>/neurons --ref-split val

  # add a memorisation check against the training split
  python tools/cloud_quality.py --clouds <gen>/clouds \\
      --ref-npy-root <bake>/neurons --ref-split val --train-split train
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.utils import farthest_point_sample_faster  # noqa: E402

# metrics/evaluation_metrics.py cannot be imported: its top-level
# `import emd_cuda` needs a CUDAExtension whose sources are absent from the release
# (plan blocker 2). We do not use EMD, so distChamfer and lgan_mmd_cov are
# replicated here verbatim from that file (:11 and :130) to keep the numbers
# directly comparable to the paper's Table 2.


def dist_chamfer(x: torch.Tensor, y: torch.Tensor):
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    di = torch.arange(0, x.size(1)).long()
    rx = xx[:, di, di].unsqueeze(1).expand_as(xx)
    ry = yy[:, di, di].unsqueeze(1).expand_as(yy)
    P = rx.transpose(2, 1) + ry - 2 * zz
    return P.min(1)[0], P.min(2)[0]


def lgan_mmd_cov(all_dist: torch.Tensor):
    _, n_ref = all_dist.size()
    min_val_fromsmp, min_idx = torch.min(all_dist, dim=1)
    min_val, _ = torch.min(all_dist, dim=0)
    return min_val.mean().item(), float(min_idx.unique().numel()) / float(n_ref)


def pairwise_cd(ref: torch.Tensor, smp: torch.Tensor) -> torch.Tensor:
    """(n_ref, n_smp) Chamfer matrix, one reference cloud at a time to bound memory."""
    M = torch.zeros(len(ref), len(smp))
    for i in range(len(ref)):
        r = ref[i:i + 1].expand(len(smp), -1, -1).contiguous()
        dl, dr = dist_chamfer(r, smp)
        M[i] = dl.mean(1) + dr.mean(1)
    return M


def unit_sphere(P: np.ndarray) -> np.ndarray:
    P = P - P.mean(axis=0)
    return P / np.sqrt((P ** 2).sum(axis=1)).max()


def gen_radius(P: np.ndarray) -> float:
    return float(np.sqrt(((P - P.mean(axis=0)) ** 2).sum(axis=1)).max())


def linearity(P: np.ndarray, k: int = 12, stride: int = 8) -> tuple[float, float]:
    """Mean local-neighbourhood linearity and median nearest-neighbour spacing.

    A neuron is filamentary: inside a small ball its points lie on a tube, so the
    local covariance has one dominant eigenvalue and lambda1/sum(lambda) -> 1. A
    diffuse or noisy cloud has more balanced eigenvalues (an isotropic blob gives
    1/3). Computed on the unit sphere so the measure is scale-free and a cloud with
    the wrong scale is still comparable.
    """
    P = unit_sphere(P)
    tree = cKDTree(P)
    dist, idx = tree.query(P, k=k + 1)
    lin = []
    for nb in idx[::stride]:
        Q = P[nb] - P[nb].mean(axis=0)
        w = np.sort(np.linalg.eigvalsh(np.cov(Q.T)))[::-1]
        lin.append(w[0] / max(w.sum(), 1e-15))
    return float(np.mean(lin)), float(np.median(dist[:, 1]))


def load_generated(d: Path) -> tuple[list[np.ndarray], list[str]]:
    files = sorted(glob.glob(os.path.join(d, '*.npy')))
    if not files:
        raise SystemExit(f'no .npy clouds under {d} -- run --stage sample first')
    return [np.load(f).astype(np.float64) for f in files], [os.path.basename(f) for f in files]


def load_reference(npy_root: Path, split: str, npoints: int, limit: int,
                   seed: int) -> list[np.ndarray]:
    """Reference clouds built exactly the way the dataloader builds them.

    `Uniform15KPC` keeps `all_points[:, 10000:]` as the test points and then draws a
    random `tr_sample_size` subset (datasets/shapenet_data_pc.py:174). Reproducing
    that here matters: the 15,000-point bake is FPS-ordered, so the last 5,000 are
    the density infill and have different local statistics from a uniform draw.
    """
    files = sorted(glob.glob(os.path.join(npy_root, split, '*.npy')))
    if not files:
        raise SystemExit(f'no .npy under {npy_root}/{split}')
    if limit:
        files = files[:limit]
    rng = np.random.default_rng(seed)
    out = []
    for f in files:
        cloud = np.load(f).astype(np.float64)
        tail = cloud[10000:]
        out.append(tail[rng.choice(len(tail), npoints, replace=False)])
    return out


def to_tensor(clouds: list[np.ndarray], npoints: int, seed: int) -> torch.Tensor:
    """Unit-normalise and FPS-downsample, matching the paper's evaluation protocol."""
    out = []
    for P in clouds:
        P = unit_sphere(P)
        if npoints < len(P):
            np.random.seed(seed)          # farthest_point_sample_faster picks its start
            P = farthest_point_sample_faster(P, npoints)   # from the numpy global rng
        out.append(P)
    return torch.tensor(np.stack(out), dtype=torch.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--clouds', type=Path, required=True,
                    help='directory of generated .npy clouds (<gen_dir>/clouds)')
    ap.add_argument('--ref-npy-root', type=Path, default=None,
                    help='the SYNSET dir of the bake, e.g. <bake>/neurons')
    ap.add_argument('--ref-split', default='val')
    ap.add_argument('--train-split', default=None,
                    help='enable the memorisation check against this split (e.g. train)')
    ap.add_argument('--npoints', type=int, default=2048)
    ap.add_argument('--cd-points', type=int, nargs='+', default=[128, 2048],
                    help='resolutions for MMD-CD/COV-CD. 128 is what the paper uses.')
    ap.add_argument('--limit', type=int, default=0,
                    help='cap the number of reference clouds (0 = all)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--output-json', type=Path, default=None)
    args = ap.parse_args()

    gen, names = load_generated(args.clouds)
    print(f'generated clouds : {len(gen)} from {args.clouds}')

    radii = np.array([gen_radius(P) for P in gen])
    lin = np.array([linearity(P) for P in gen])
    out = {
        'n_generated': len(gen),
        'gen_radius': {'median': float(np.median(radii)),
                       'p10': float(np.percentile(radii, 10)),
                       'p90': float(np.percentile(radii, 90))},
        'linearity': float(lin[:, 0].mean()),
        'nn_spacing': float(lin[:, 1].mean()),
    }
    print('\n=== cloud health (no reconstruction) ===')
    print('  gen_radius        median %.3g  p10 %.3g  p90 %.3g   (training target: exactly 1.0)'
          % (out['gen_radius']['median'], out['gen_radius']['p10'], out['gen_radius']['p90']))
    print('  linearity         %.3f   (1.0 = points on a tube, 0.333 = isotropic blob)'
          % out['linearity'])
    print('  NN spacing        %.4f  (unit sphere)' % out['nn_spacing'])
    if not 0.5 < out['gen_radius']['median'] < 2.0:
        print('  WARNING: far out of distribution. detect_radius and length_threshold are '
              'unit-sphere constants, so any morphometric from these clouds is an artifact '
              'of the wrong scale. Reject this checkpoint before reconstructing it.')

    if args.ref_npy_root:
        ref = load_reference(args.ref_npy_root, args.ref_split, args.npoints,
                             args.limit, args.seed)
        rlin = np.array([linearity(P) for P in ref])
        out['reference'] = {'split': args.ref_split, 'n': len(ref),
                            'linearity': float(rlin[:, 0].mean()),
                            'nn_spacing': float(rlin[:, 1].mean())}
        print('\n=== reference (%s, n=%d) ===' % (args.ref_split, len(ref)))
        print('  linearity         %.3f' % out['reference']['linearity'])
        print('  NN spacing        %.4f' % out['reference']['nn_spacing'])
        print('  generated retains %.0f%% of reference linearity'
              % (100 * out['linearity'] / max(out['reference']['linearity'], 1e-9)))

        out['cd'] = {}
        print('\n=== MorphoGen\'s own metric (paper Tab. 2 uses 128 points) ===')
        for npts in args.cd_points:
            R = to_tensor(ref, npts, args.seed)
            G = to_tensor(gen, npts, args.seed)
            mmd, cov = lgan_mmd_cov(pairwise_cd(R, G).t())
            out['cd'][str(npts)] = {'mmd_cd': mmd, 'cov_cd': cov}
            print('  %4d points        MMD-CD %.5f   COV-CD %.3f%s'
                  % (npts, mmd, cov, '   <- paper' if npts == 128 else ''))
        print('  paper Tab. 2 @128 : MMD-CD 0.0324 / 0.0162 / 0.0201 (IT / CT / PT)')
        print('  NOTE: MMD and COV are N-dependent; only compare runs at equal N.')

        if args.train_split:
            # Memorisation check. 690 neurons x 10,150 passes invites recall, and
            # MMD/COV cannot detect it: a model that reproduces training neurons
            # scores WELL on both. The test is whether generated clouds sit closer
            # to the training set than held-out real neurons do.
            tr = load_reference(args.ref_npy_root, args.train_split, args.npoints,
                                args.limit or len(ref), args.seed)
            npts = min(args.cd_points)
            T = to_tensor(tr, npts, args.seed)
            g_nn = pairwise_cd(T, to_tensor(gen, npts, args.seed)).min(0)[0]
            r_nn = pairwise_cd(T, to_tensor(ref, npts, args.seed)).min(0)[0]
            g_med, r_med = float(g_nn.median()), float(r_nn.median())
            out['memorisation'] = {
                'points': npts, 'n_train': len(tr),
                'gen_to_train_median': g_med,
                'heldout_to_train_median': r_med,
            }
            print('\n=== memorisation check (@%d points, %d train clouds) ==='
                  % (npts, len(tr)))
            print('  generated -> nearest train   median CD %.5f' % g_med)
            print('  held-out  -> nearest train   median CD %.5f' % r_med)
            if r_med < 1e-6:
                # Not a memorisation result: a held-out neuron sitting at zero Chamfer
                # from a training neuron means the two splits share files. That is a
                # corpus bug and it invalidates every held-out number, so say so
                # instead of printing a meaningless ratio.
                out['memorisation']['split_overlap'] = True
                print('  SPLIT OVERLAP: held-out clouds are at ~zero Chamfer from training '
                      'clouds, i.e. %s and %s share neurons. Fix the corpus; the '
                      'memorisation check (and every held-out metric) is meaningless '
                      'until then.' % (args.ref_split, args.train_split))
            else:
                ratio = g_med / r_med
                out['memorisation']['ratio'] = ratio
                print('  ratio %.2f   (<1 means generated clouds are CLOSER to the '
                      'training set than real held-out neurons are, i.e. recall rather '
                      'than generation)' % ratio)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(out, indent=2, default=float))
        print(f'\nwrote {args.output_json}')


if __name__ == '__main__':
    main()
