import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class BatchInfo:
    anchor_camera_id: int
    batch_camera_ids: List[int]       # B camera indices (anchor + neighbours)
    unrelated_batches: List[int]      # anchor indices of unrelated batches
    batch_blocks: np.ndarray          # int32 unique voxel block IDs for this batch (~KB)
    has_dirty_overlap: bool           # True if no perfectly disjoint partner exists
    n_gaussians: int                  # approx Gaussian count (sum of block sizes)


class CameraBatchScheduler:
    """
    Builds the full precomputation schedule for Stage 2 dynamic Gaussian loading.

    Steps:
      3. Pairwise IoU on CPU (ThreadPoolExecutor) → knn_table (N, B)
      4. Per-batch Gaussian union (downsampled bool masks, bitwise OR)
      5. Non-overlapping batch pairing
      6. Block ID assignment per batch (from ds_blocks + visible_ds)
      7. Assemble BatchInfo per anchor camera
    """

    def __init__(
        self,
        cameras: List[Any],
        visible_ds: np.ndarray,      # (N, M) bool — output of VisibilityPrecomputer
        batch_size: int = 8,
        n_workers: Optional[int] = None,
    ):
        self.cameras   = cameras
        self.visible_ds = visible_ds  # (N, M) bool on CPU
        self.N  = len(cameras)
        self.M  = visible_ds.shape[1]
        self.B  = batch_size
        self.n_workers = n_workers or max(1, os.cpu_count())

    # ------------------------------------------------------------------
    # Step 3: KNN table via pairwise IoU
    # ------------------------------------------------------------------
    def build_knn_table(self) -> np.ndarray:
        """
        Compute pairwise camera IoU in parallel, return knn_table (N, B).
        knn_table[i] = [i, neighbour_1, ..., neighbour_{B-1}]  (all camera indices)
        """
        N, B = self.N, self.B
        iou = np.zeros((N, N), dtype=np.float32)

        pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]

        def _iou(pair):
            i, j = pair
            a, b  = self.visible_ds[i], self.visible_ds[j]
            inter = np.count_nonzero(a & b)
            union = np.count_nonzero(a | b)
            return i, j, inter / union if union > 0 else 0.0

        print(f"  [CameraBatchScheduler] Computing {len(pairs)} pairwise IoUs "
              f"with {self.n_workers} workers...")
        with ThreadPoolExecutor(max_workers=self.n_workers) as pool:
            for i, j, val in pool.map(_iou, pairs):
                iou[i, j] = val
                iou[j, i] = val

        knn_table = np.zeros((N, B), dtype=np.int64)
        for i in range(N):
            row = iou[i].copy()
            row[i] = -1.0                              # exclude self
            top = np.argsort(row)[::-1][: B - 1]
            knn_table[i, 0]  = i
            knn_table[i, 1:] = top

        return knn_table

    # ------------------------------------------------------------------
    # Step 4: Per-batch Gaussian union (downsampled, bool OR)
    # ------------------------------------------------------------------
    def build_batch_gaussians_ds(self, knn_table: np.ndarray) -> np.ndarray:
        """
        Returns batch_gaussians_ds: (N, M) bool.
        Row i = bitwise OR of visible_ds rows for all cameras in knn_table[i].
        """
        N, M = self.N, self.M
        batch_gaussians_ds = np.zeros((N, M), dtype=bool)
        for i in range(N):
            union = self.visible_ds[knn_table[i, 0]].copy()
            for cam_idx in knn_table[i, 1:]:
                union |= self.visible_ds[cam_idx]
            batch_gaussians_ds[i] = union
        return batch_gaussians_ds

    # ------------------------------------------------------------------
    # Step 5: Non-overlapping batch pairing
    # ------------------------------------------------------------------
    def build_unrelated_pairs(
        self,
        knn_table: np.ndarray,
        batch_gaussians_ds: np.ndarray,
    ) -> Dict[int, List[int]]:
        """
        Returns {anchor_idx → [list of unrelated anchor indices]}.
        Unrelated = no shared cameras AND no shared Gaussians.
        Falls back to least-overlap partner if none fully disjoint.
        """
        N = self.N
        camera_sets = [set(knn_table[i].tolist()) for i in range(N)]
        unrelated: Dict[int, List[int]] = {}

        for i in range(N):
            cs_i  = camera_sets[i]
            bg_i  = batch_gaussians_ds[i]
            clean = []

            for j in range(N):
                if i == j:
                    continue
                if cs_i & camera_sets[j]:
                    continue                             # shared camera
                if not (bg_i & batch_gaussians_ds[j]).any():
                    clean.append(j)

            if clean:
                unrelated[i] = clean
            else:
                # Fallback: camera-disjoint, least Gaussian overlap
                cam_disjoint = [
                    j for j in range(N)
                    if j != i and not (camera_sets[j] & cs_i)
                ]
                if cam_disjoint:
                    overlaps = {
                        j: int((bg_i & batch_gaussians_ds[j]).sum())
                        for j in cam_disjoint
                    }
                    best = min(overlaps, key=overlaps.get)
                else:
                    # Global fallback
                    overlaps = {
                        j: int((bg_i & batch_gaussians_ds[j]).sum())
                        for j in range(N) if j != i
                    }
                    best = min(overlaps, key=overlaps.get)
                unrelated[i] = [best]
                print(f"  [WARN] Batch {i} has no fully disjoint partner; "
                      f"using least-overlap batch {best}. "
                      f"Dirty write possible.")

        return unrelated

    # ------------------------------------------------------------------
    # Step 6: Block ID assignment per batch
    # ------------------------------------------------------------------
    def build_batch_blocks(
        self,
        knn_table: np.ndarray,
        ds_blocks: np.ndarray,          # (M,) int32 — block id of each ds point
        block_starts: np.ndarray,       # (K,) int64 from SpatialBlockIndex
        block_ends: np.ndarray,         # (K,) int64 from SpatialBlockIndex
    ) -> Dict[int, tuple]:
        """
        For each batch i: find unique block ids touched by visible downsampled points,
        then compute approximate full-scale Gaussian count from block sizes.

        Returns {batch_idx → (batch_blocks: np.ndarray int32, n_gaussians: int)}
        """
        N = self.N
        result: Dict[int, tuple] = {}
        print(f"  [CameraBatchScheduler] Step 6: assigning voxel blocks for {N} batches...")
        for i in range(N):
            visible_mask = self.visible_ds[knn_table[i]].any(axis=0)  # (M,) bool
            blocks = np.unique(ds_blocks[visible_mask].astype(np.int32))
            n_gauss = int(sum(
                block_ends[b] - block_starts[b] for b in blocks
            ))
            result[i] = (blocks, n_gauss)
            if (i + 1) % 50 == 0 or i == N - 1:
                print(f"    batch {i + 1}/{N}: {len(blocks)} blocks, "
                      f"~{n_gauss:,} Gaussians")
        return result

    # ------------------------------------------------------------------
    # Step 7: Assemble BatchInfo table
    # ------------------------------------------------------------------
    def build(
        self,
        ds_blocks: np.ndarray,
        block_index,                    # SpatialBlockIndex (avoids circular import)
        fov_margin: float = 0.1,
        slot_budget_bytes: int = 4 * 1024 ** 3,
    ) -> Dict[int, BatchInfo]:
        """
        Run all scheduler steps and return {anchor_camera_idx → BatchInfo}.

        ds_blocks   : (M,) int32 block ids for each downsampled point.
        block_index : SpatialBlockIndex — provides block_starts/block_ends.
        """
        bytes_per_gaussian = 56   # xyz(12) + f_dc(12) + scaling(12) + rot(16) + opacity(4)
        max_gaussians = slot_budget_bytes // bytes_per_gaussian

        print("[CameraBatchScheduler] Step 3: KNN table...")
        knn_table = self.build_knn_table()

        print("[CameraBatchScheduler] Step 4: per-batch downsampled Gaussian union...")
        batch_gaussians_ds = self.build_batch_gaussians_ds(knn_table)

        print("[CameraBatchScheduler] Step 5: non-overlapping batch pairs...")
        unrelated = self.build_unrelated_pairs(knn_table, batch_gaussians_ds)

        print("[CameraBatchScheduler] Step 6: voxel block assignment...")
        batch_block_map = self.build_batch_blocks(
            knn_table, ds_blocks, block_index.block_starts, block_index.block_ends
        )

        print("[CameraBatchScheduler] Step 7: assembling BatchInfo table...")
        batch_infos: Dict[int, BatchInfo] = {}
        n_over_budget = 0

        for i in range(self.N):
            cam_ids        = knn_table[i].tolist()
            blocks, n_gauss = batch_block_map[i]
            is_dirty       = i not in unrelated or not unrelated[i]

            if n_gauss > max_gaussians:
                n_over_budget += 1
                print(f"  [WARN] Batch {i}: ~{n_gauss:,} Gaussians > budget "
                      f"{max_gaussians:,}. Will be skipped at training time.")

            batch_infos[i] = BatchInfo(
                anchor_camera_id=i,
                batch_camera_ids=cam_ids,
                unrelated_batches=unrelated.get(i, []),
                batch_blocks=blocks,
                has_dirty_overlap=is_dirty,
                n_gaussians=n_gauss,
            )

        if n_over_budget:
            print(f"[CameraBatchScheduler] {n_over_budget}/{self.N} batches exceed "
                  f"the GPU slot budget and will be skipped during training.")

        print("[CameraBatchScheduler] Done.")
        return batch_infos
