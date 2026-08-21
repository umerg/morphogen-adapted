"""Trilinear devoxelization.

Pure-PyTorch trilinear sampling of a voxel grid at continuous point
coordinates, equivalent to the PVCNN CUDA kernel
(``interpolate/trilinear_devox.cu``) but autograd-native, so no custom backward
is needed.  Runs on CPU, MPS and CUDA.

Flat voxel index follows the upstream convention ``x * r^2 + y * r + z``.
"""

import torch

from modules.functional.backend import _backend

__all__ = ['trilinear_devoxelize']


def _trilinear_devoxelize_torch(features, coords, resolution):
    """
    :param features: FloatTensor[B, C, R, R, R] (or [B, C, R^3])
    :param coords:   FloatTensor[B, 3, N], continuous coords in [0, r-1]
    :param resolution: int, voxel resolution r
    :return: FloatTensor[B, C, N]
    """
    b, c = features.shape[:2]
    r = int(resolution)
    n = coords.shape[-1]

    feats = features.contiguous().view(b, c, r * r * r)

    coords = coords.clamp(0, r - 1)
    lo = torch.floor(coords)
    # Upper corner is the next voxel, clamped so points on the far face stay in range.
    hi = (lo + 1).clamp(max=r - 1)
    frac = coords - lo                                    # [B, 3, N] in [0, 1]

    lo = lo.long()
    hi = hi.long()
    fx, fy, fz = frac[:, 0], frac[:, 1], frac[:, 2]       # each [B, N]

    out = features.new_zeros(b, c, n)
    # Accumulate the eight corners with their trilinear weights.
    for dx in (0, 1):
        ix = hi[:, 0] if dx else lo[:, 0]
        wx = fx if dx else (1.0 - fx)
        for dy in (0, 1):
            iy = hi[:, 1] if dy else lo[:, 1]
            wy = fy if dy else (1.0 - fy)
            for dz in (0, 1):
                iz = hi[:, 2] if dz else lo[:, 2]
                wz = fz if dz else (1.0 - fz)

                idx = ix * (r * r) + iy * r + iz          # [B, N]
                gathered = torch.gather(feats, 2, idx.unsqueeze(1).expand(b, c, n))
                out = out + gathered * (wx * wy * wz).unsqueeze(1)

    return out


class TrilinearDevoxelization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, coords, resolution, is_training=True):
        """
        :param ctx:
        :param coords: the coordinates of points, FloatTensor[B, 3, N]
        :param features: FloatTensor[B, C, R, R, R]
        :param resolution: int, the voxel resolution
        :param is_training: bool, training mode
        :return:
            FloatTensor[B, C, N]
        """
        B, C = features.shape[:2]
        features = features.contiguous().view(B, C, -1)
        coords = coords.contiguous()
        outs, inds, wgts = _backend.trilinear_devoxelize_forward(resolution, is_training, coords, features)
        if is_training:
            ctx.save_for_backward(inds, wgts)
            ctx.r = resolution
        return outs

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx:
        :param grad_output: gradient of outputs, FloatTensor[B, C, N]
        :return:
            gradient of inputs, FloatTensor[B, C, R, R, R]
        """
        inds, wgts = ctx.saved_tensors
        grad_inputs = _backend.trilinear_devoxelize_backward(grad_output.contiguous(), inds, wgts, ctx.r)
        return grad_inputs.view(grad_output.size(0), grad_output.size(1), ctx.r, ctx.r, ctx.r), None, None, None


def trilinear_devoxelize(features, coords, resolution, is_training=True):
    """Dispatch to the compiled kernel when available, else pure PyTorch."""
    if _backend is not None and features.is_cuda:
        return TrilinearDevoxelization.apply(features, coords, resolution, is_training)
    return _trilinear_devoxelize_torch(features, coords, resolution)
