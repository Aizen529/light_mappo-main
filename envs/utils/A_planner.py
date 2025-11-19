from __future__ import annotations

import heapq
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

Coord = Tuple[int, int]


class AStarPlanner:
    """Grid-based A* planner with configurable neighbor connectivity."""

    def __init__(self, neighbor_mode: str = "4"):
        """
        Args:
            neighbor_mode: "4" for von Neumann (cardinal directions) or
                "8" for Moore (includes diagonals) connectivity.
        """
        if neighbor_mode not in {"4", "8"}:
            raise ValueError("neighbor_mode must be either '4' or '8'.")
        self.neighbor_mode = neighbor_mode
        self._neighbor_offsets = self._build_neighbor_offsets(neighbor_mode)

    def plan(
        self,
        start: Coord,
        goal: Coord,
        global_map: Sequence[Sequence[int]],
    ) -> List[Coord]:
        """
        Plan a collision-free path between two coordinates on a grid.

        Args:
            start: (row, col) tuple of the robot's location.
            goal: (row, col) tuple of the desired goal.
            global_map: occupancy grid where values < 0 denote obstacles.

        Returns:
            List of coordinates from start to goal (inclusive). Empty if unreachable.
        """
        grid = np.asarray(global_map)
        if grid.ndim != 2:
            raise ValueError("global_map must be a 2-D array.")

        if not self._is_walkable(start, grid) or not self._is_walkable(goal, grid):
            return []

        if start == goal:
            return [start]

        open_heap: List[Tuple[float, int, Coord]] = []
        heapq.heappush(open_heap, (self._heuristic(start, goal), 0, start))

        came_from: dict[Coord, Coord] = {}
        g_score = {start: 0.0}

        expansions = 0
        while open_heap:
            _, _, current = heapq.heappop(open_heap)

            if current == goal:
                return self._reconstruct_path(came_from, current)

            for delta in self._neighbor_offsets:
                neighbor = (current[0] + delta[0], current[1] + delta[1])
                if not self._is_walkable(neighbor, grid):
                    continue

                tentative_g = g_score[current] + 1.0
                if tentative_g >= g_score.get(neighbor, math.inf):
                    continue

                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                priority = tentative_g + self._heuristic(neighbor, goal)
                expansions += 1
                heapq.heappush(open_heap, (priority, expansions, neighbor))

        return []

    @staticmethod
    def _build_neighbor_offsets(mode: str) -> List[Coord]:
        cardinal = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        if mode == "4":
            return cardinal
        return cardinal + [(1, 1), (1, -1), (-1, 1), (-1, -1)]

    @staticmethod
    def _heuristic(a: Coord, b: Coord) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def _is_walkable(coord: Coord, grid: np.ndarray) -> bool:
        row, col = coord
        if row < 0 or row >= grid.shape[0] or col < 0 or col >= grid.shape[1]:
            return False
        return grid[row, col] >= 0

    @staticmethod
    def _reconstruct_path(came_from: dict[Coord, Coord], current: Coord) -> List[Coord]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
