#!/usr/bin/env python3
"""
Convert cloud_merged.las → voxel-downsampled PLY in 3DGS-compatible format.

Usage:
    python las_to_ply.py <input.las> <output.ply> [voxel_size]

Output PLY vertex fields: x y z nx ny nz red green blue
(matches the format expected by scene/dataset_readers.py::fetchPly / storePly)
"""

import sys
import numpy as np
import laspy
from plyfile import PlyData, PlyElement


def _ravel_hash(arr: np.ndarray) -> np.ndarray:
    """Fortran-order hash for integer coordinate rows (same as train_stage2.py)."""
    assert arr.ndim == 2
    arr = arr.copy()
    arr -= arr.min(0)
    arr = arr.astype(np.uint64, copy=False)
    arr_max = arr.max(0).astype(np.uint64) + 1
    keys = np.zeros(arr.shape[0], dtype=np.uint64)
    for j in range(arr.shape[1] - 1):
        keys += arr[:, j]
        keys *= arr_max[j + 1]
    keys += arr[:, -1]
    return keys


def voxel_downsample(xyz: np.ndarray, voxel_size: float) -> np.ndarray:
    """
    Voxel-grid downsample: one random point kept per occupied voxel.
    Returns index array into xyz (same algorithm as train_stage2.py).
    """
    discrete = np.floor(xyz / voxel_size).astype(np.int64)
    key = _ravel_hash(discrete)
    idx_sort = np.argsort(key)
    key_sort = key[idx_sort]
    _, _, count = np.unique(key_sort, return_inverse=True, return_counts=True)
    idx_select = (
        np.cumsum(np.insert(count, 0, 0)[:-1])
        + np.random.randint(0, count.max(), count.size) % count
    )
    return idx_sort[idx_select]


def las_to_ply(las_path: str, ply_path: str, voxel_size: float = 0.08) -> int:
    print(f"[las_to_ply] Reading {las_path} ...")
    las = laspy.read(las_path)

    xyz = np.stack([
        np.asarray(las.x, dtype=np.float64),
        np.asarray(las.y, dtype=np.float64),
        np.asarray(las.z, dtype=np.float64),
    ], axis=1)

    # LAS colours are uint16 (0–65535); normalise to uint8 (0–255)
    r = np.asarray(las.red,   dtype=np.uint16)
    g = np.asarray(las.green, dtype=np.uint16)
    b = np.asarray(las.blue,  dtype=np.uint16)
    if r.max() > 255:
        r = (r >> 8).astype(np.uint8)
        g = (g >> 8).astype(np.uint8)
        b = (b >> 8).astype(np.uint8)
    else:
        r = r.astype(np.uint8)
        g = g.astype(np.uint8)
        b = b.astype(np.uint8)

    print(f"[las_to_ply]   Raw points : {len(xyz):>12,}")

    idx = voxel_downsample(xyz, voxel_size)
    xyz = xyz[idx]
    r, g, b = r[idx], g[idx], b[idx]
    N = len(xyz)

    print(f"[las_to_ply]   After voxel downsample ({voxel_size} m): {N:>12,}")

    # Build PLY vertex array (3DGS storePly-compatible layout)
    dtype = [
        ('x',  'f4'), ('y',  'f4'), ('z',  'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
    ]
    vertex = np.empty(N, dtype=dtype)
    vertex['x']  = xyz[:, 0].astype('f4')
    vertex['y']  = xyz[:, 1].astype('f4')
    vertex['z']  = xyz[:, 2].astype('f4')
    vertex['nx'] = vertex['ny'] = vertex['nz'] = np.float32(0.0)
    vertex['red']   = r
    vertex['green'] = g
    vertex['blue']  = b

    PlyData([PlyElement.describe(vertex, 'vertex')]).write(ply_path)
    print(f"[las_to_ply]   Saved → {ply_path}")
    return N


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: las_to_ply.py <input.las> <output.ply> [voxel_size=0.08]")
        sys.exit(1)

    _las  = sys.argv[1]
    _ply  = sys.argv[2]
    _vox  = float(sys.argv[3]) if len(sys.argv) > 3 else 0.08
    las_to_ply(_las, _ply, _vox)
