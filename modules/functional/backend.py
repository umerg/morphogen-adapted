"""PVCNN CUDA extension loader.

The released repository does not ship ``modules/functional/src`` (no .cpp/.cu
sources), so the JIT build below cannot succeed as-is.  Importing this module
used to raise at *import* time, which made ``import DDPM_train`` fail on every
machine, CUDA or not.

We therefore load the extension opportunistically and fall back to ``None``.
Every consumer in this package touches ``_backend`` only inside a function
body, so a ``None`` backend costs nothing until one of the CUDA-only ops is
actually called -- and the two ops DiT-3D needs (``avg_voxelize`` and
``trilinear_devoxelize``) have pure-PyTorch implementations that do not use
this backend at all.

To restore the compiled path, vendor the upstream DiT-3D / PVCNN ``src``
directory next to this file; it will then be picked up automatically.
"""

import os
import warnings

_src_path = os.path.dirname(os.path.abspath(__file__))

_SOURCES = [
    'ball_query/ball_query.cpp',
    'ball_query/ball_query.cu',
    'grouping/grouping.cpp',
    'grouping/grouping.cu',
    'interpolate/neighbor_interpolate.cpp',
    'interpolate/neighbor_interpolate.cu',
    'interpolate/trilinear_devox.cpp',
    'interpolate/trilinear_devox.cu',
    'sampling/sampling.cpp',
    'sampling/sampling.cu',
    'voxelization/vox.cpp',
    'voxelization/vox.cu',
    'bindings.cpp',
]

_backend = None
_backend_error = None


def _try_load():
    """Return the compiled extension, or None if it cannot be built."""
    sources = [os.path.join(_src_path, 'src', f) for f in _SOURCES]
    missing = [s for s in sources if not os.path.isfile(s)]
    if missing:
        return None, FileNotFoundError(
            'PVCNN CUDA sources are absent from this checkout (e.g. {}). '
            'Vendor them from https://github.com/DiT-3D/DiT-3D to enable the '
            'compiled path.'.format(os.path.relpath(missing[0], _src_path))
        )

    try:
        import torch
        from torch.utils.cpp_extension import load
    except Exception as exc:  # pragma: no cover - torch is a hard dependency
        return None, exc

    if not torch.cuda.is_available():
        return None, RuntimeError('CUDA is not available; skipping PVCNN extension build.')

    try:
        ext = load(
            name='_pvcnn_backend',
            extra_cflags=['-O3', '-std=c++17'],
            extra_cuda_cflags=[],
            sources=sources,
        )
        return ext, None
    except Exception as exc:
        return None, exc


_backend, _backend_error = _try_load()

if _backend is None:
    warnings.warn(
        'PVCNN CUDA backend unavailable ({}). avg_voxelize and '
        'trilinear_devoxelize will use the pure-PyTorch implementations; the '
        'remaining ops in modules.functional will raise if called.'.format(_backend_error),
        RuntimeWarning,
        stacklevel=2,
    )


__all__ = ['_backend', '_backend_error']
