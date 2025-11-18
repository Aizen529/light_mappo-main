# subgoal.py
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage


@dataclass
class Subgoal:
    """
    子目标（未覆盖连通区域）的描述。所有坐标使用 (row, col) = (y, x) 约定。
    """

    id: int
    centroid: Tuple[float, float]
    mask: np.ndarray
    area: int
    bbox: Tuple[int, int, int, int]
    extra: Optional[Dict] = None


class SubgoalGenerator:
    """
    从单通道全局状态图生成子目标。

    global_grid 编码:
        -1: 障碍
         0: 未覆盖可通行区域
         1: 已覆盖可通行区域
    """

    def __init__(
        self,
        min_area: int = 1,
        connectivity: int = 8,
        sort_by_area: bool = False,
    ):
        if min_area < 1:
            raise ValueError("min_area must be >= 1")
        if connectivity not in (4, 8):
            raise ValueError("connectivity must be 4 or 8")

        self.min_area = min_area
        self.connectivity = connectivity
        self.sort_by_area = sort_by_area
        self._structure = self._create_connectivity_structure(connectivity)

    def __call__(self, global_grid: np.ndarray) -> List[Subgoal]:
        return self.generate(global_grid)

    def generate(self, global_grid: np.ndarray) -> List[Subgoal]:
        """
        输入:
            global_grid: shape = (H, W) 的单通道整数网格
        返回:
            List[Subgoal]
        """
        validated_grid = self._validate_global_grid(global_grid)
        uncovered_mask = self._build_uncovered_mask(validated_grid)

        if not uncovered_mask.any():
            return []

        labels, num_labels = ndimage.label(
            uncovered_mask, structure=self._structure
        )

        if num_labels == 0:
            return []

        subgoals: List[Subgoal] = []

        for label_idx in range(1, num_labels + 1):
            coords = np.argwhere(labels == label_idx)
            area = coords.shape[0]
            if area < self.min_area:
                continue

            rows = coords[:, 0]
            cols = coords[:, 1]
            centroid = (float(rows.mean()), float(cols.mean()))

            region_mask = np.zeros_like(uncovered_mask, dtype=bool)
            region_mask[rows, cols] = True

            min_row = int(rows.min())
            max_row = int(rows.max())
            min_col = int(cols.min())
            max_col = int(cols.max())
            bbox = (min_row, min_col, max_row, max_col)

            subgoals.append(
                Subgoal(
                    id=len(subgoals),
                    centroid=centroid,
                    mask=region_mask,
                    area=int(area),
                    bbox=bbox,
                    extra={},
                )
            )

        if self.sort_by_area:
            subgoals.sort(key=lambda sg: sg.area, reverse=True)

        return subgoals

    @staticmethod
    def _validate_global_grid(global_grid: np.ndarray) -> np.ndarray:
        if global_grid.ndim != 2:
            raise ValueError("global_grid must be a 2D array")

        if not np.issubdtype(global_grid.dtype, np.integer):
            return global_grid.astype(np.int32)

        return global_grid

    @staticmethod
    def _build_uncovered_mask(global_grid: np.ndarray) -> np.ndarray:
        return global_grid == 0

    @staticmethod
    def _create_connectivity_structure(connectivity: int) -> np.ndarray:
        order = 2 if connectivity == 8 else 1
        return ndimage.generate_binary_structure(rank=2, connectivity=order)


def generate_subgoals(
    global_grid: np.ndarray,
    min_area: int = 1,
    connectivity: int = 8,
    sort_by_area: bool = False,
) -> List[Subgoal]:
    """
    便捷函数：一次性创建生成器并返回子目标列表。
    """

    generator = SubgoalGenerator(
        min_area=min_area,
        connectivity=connectivity,
        sort_by_area=sort_by_area,
    )
    return generator.generate(global_grid)


if __name__ == "__main__":
    global_grid = np.array(
        [
            [0, 0, -1, -1, 0],
            [0, 1, 1, -1, 0],
            [0, 0, 0, 0, 0],
            [-1, -1, 0, 1, 1],
        ],
        dtype=np.int8,
    )

    generator = SubgoalGenerator(min_area=2, connectivity=8, sort_by_area=False)
    subgoals = generator.generate(global_grid)

    print(f"num subgoals: {len(subgoals)}")
    for sg in subgoals:
        print(
            f"ID={sg.id}, area={sg.area}, centroid={sg.centroid}, bbox={sg.bbox}"
        )
