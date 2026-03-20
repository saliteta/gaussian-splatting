import numpy as np


class SpatialBlockIndex:
    """
    Voxel-grid spatial index for fast block-based Gaussian lookup.

    Divides the bounding box into resolution^3 cells, sorts all points once
    by cell id, and stores block_starts/block_ends for O(1) range lookup.

    Memory: ~12 bytes/point (spatial_order int64) + 2*K int64 for blocks.
    Replaces the ~81 GB gaussian_indices dict (506 batches × 20M × 8 bytes).
    """

    def __init__(self, xyz_full: np.ndarray, resolution: int = 10):
        """
        xyz_full : (P, 3) float32 numpy array — full-scale Gaussian positions.
        resolution: voxel grid side length; total cells K = resolution^3.
        """
        P = xyz_full.shape[0]
        R = resolution
        K = R * R * R

        # 1. Bounding box with small pad to avoid edge cases
        mn = xyz_full.min(axis=0)
        mx = xyz_full.max(axis=0)
        pad = (mx - mn) * 0.001 + 1e-6
        mn = mn - pad
        mx = mx + pad
        self._mn = mn
        self._mx = mx
        self._R  = R

        # 2. cell_id[i] = ix*R*R + iy*R + iz  (0 .. K-1)
        norm = (xyz_full - mn) / (mx - mn)          # (P, 3) in [0, 1)
        norm = np.clip(norm, 0.0, 1.0 - 1e-7)
        ix = (norm[:, 0] * R).astype(np.int32)
        iy = (norm[:, 1] * R).astype(np.int32)
        iz = (norm[:, 2] * R).astype(np.int32)
        cell_id = (ix * (R * R) + iy * R + iz)      # (P,) int32

        # 3. Sort by cell id (stable) → spatial_order : (P,) int64
        self.spatial_order = np.argsort(cell_id, kind='stable').astype(np.int64)

        # 4. Sorted cell ids after reorder
        sorted_cells = cell_id[self.spatial_order]

        # 5. block_starts[b] / block_ends[b] — start/end in reordered store
        self.block_starts = np.zeros(K, dtype=np.int64)
        self.block_ends   = np.zeros(K, dtype=np.int64)

        if P > 0:
            change_mask  = np.concatenate([[True], sorted_cells[1:] != sorted_cells[:-1]])
            unique_cells = sorted_cells[change_mask]
            starts       = np.where(change_mask)[0].astype(np.int64)
            ends         = np.concatenate([starts[1:], [P]]).astype(np.int64)
            self.block_starts[unique_cells] = starts
            self.block_ends[unique_cells]   = ends

        print(f"[SpatialBlockIndex] {P:,} points → {K} voxel cells "
              f"(resolution={R}^3), spatial_order built.")

    def assign_blocks(self, xyz_query: np.ndarray) -> np.ndarray:
        """
        xyz_query : (M, 3) float32 — e.g. downsampled point cloud.
        Returns   : (M,) int32 block ids using the same voxel grid.
        Points outside bbox are clamped to the nearest cell.
        """
        R  = self._R
        mn = self._mn
        mx = self._mx
        norm = (xyz_query - mn) / (mx - mn)
        norm = np.clip(norm, 0.0, 1.0 - 1e-7)
        ix = (norm[:, 0] * R).astype(np.int32)
        iy = (norm[:, 1] * R).astype(np.int32)
        iz = (norm[:, 2] * R).astype(np.int32)
        return (ix * (R * R) + iy * R + iz).astype(np.int32)

    def get_block_indices(self, block_ids: np.ndarray) -> np.ndarray:
        """
        Expand block_ids → flat sorted int64 index array into the reordered store.

        block_ids : (B,) int32/int64 unique block IDs.
        Returns   : (S,) int64 contiguous indices into cpu_store (after reorder).

        For each b: np.arange(block_starts[b], block_ends[b]) → concatenate.
        Empty blocks (no Gaussians in that voxel) are silently skipped.
        """
        if len(block_ids) == 0:
            return np.empty(0, dtype=np.int64)

        parts = []
        for b in block_ids:
            s = self.block_starts[b]
            e = self.block_ends[b]
            if e > s:
                parts.append(np.arange(s, e, dtype=np.int64))

        if not parts:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(parts)
