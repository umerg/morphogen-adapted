"""Average voxelization.

Pure-PyTorch scatter-mean, numerically equivalent to the PVCNN CUDA kernel
(``voxelization/vox.cu``) but written with autograd-native ops so no custom
backward is needed.  Runs on CPU, MPS and CUDA.

Flat voxel index follows the upstream convention ``x * r^2 + y * r + z``, which
is what ``out.view(b, c, r, r, r)`` in the original implementation assumes.
"""

import torch

from modules.functional.backend import _backend

__all__ = ['avg_voxelize']


def _avg_voxelize_torch(features, coords, resolution):
    """
    :param features: FloatTensor[B, C, N]
    :param coords:   IntTensor[B, 3, N], integer voxel coordinates in [0, r-1]
    :param resolution: int, voxel resolution r
    :return: FloatTensor[B, C, r, r, r]
    """
    b, c, n = features.shape
    r = int(resolution)
    r3 = r * r * r

    coords = coords.long().clamp_(0, r - 1)
    # flat index per point, upstream layout: x * r^2 + y * r + z
    idx = coords[:, 0] * (r * r) + coords[:, 1] * r + coords[:, 2]   # [B, N]

    # Sum features into their voxel, and count points per voxel.
    idx_e = idx.unsqueeze(1).expand(b, c, n)                          # [B, C, N]
    out = features.new_zeros(b, c, r3).scatter_add_(2, idx_e, features)

    ones = features.new_ones(b, 1, n)
    counts = features.new_zeros(b, 1, r3).scatter_add_(2, idx.unsqueeze(1), ones)

    # Empty voxels stay exactly zero (matches the CUDA kernel).
    out = out / counts.clamp(min=1.0)
    return out.view(b, c, r, r, r)


class AvgVoxelization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, coords, resolution):
        """
        :param ctx:
        :param features: Features of the point cloud, FloatTensor[B, C, N]
        :param coords: Voxelized Coordinates of each point, IntTensor[B, 3, N]
        :param resolution: Voxel resolution
        :return:
            Voxelized Features, FloatTensor[B, C, R, R, R]
        """
        features = features.contiguous()
        coords = coords.int().contiguous()
        b, c, _ = features.shape
        out, indices, counts = _backend.avg_voxelize_forward(features, coords, resolution)
        ctx.save_for_backward(indices, counts)
        return out.view(b, c, resolution, resolution, resolution)

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx:
        :param grad_output: gradient of output, FloatTensor[B, C, R, R, R]
        :return:
            gradient of inputs, FloatTensor[B, C, N]
        """
        b, c = grad_output.shape[:2]
        indices, counts = ctx.saved_tensors
        grad_features = _backend.avg_voxelize_backward(grad_output.contiguous().view(b, c, -1), indices, counts)
        return grad_features, None, None


def avg_voxelize(features, coords, resolution):
    """Dispatch to the compiled kernel when available, else pure PyTorch."""
    if _backend is not None and features.is_cuda:
        return AvgVoxelization.apply(features, coords, resolution)
    return _avg_voxelize_torch(features, coords, resolution)
