from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .A_planner import AStarPlanner

Coord = Tuple[int, int]


@dataclass
class AssignmentResult:
    """Container for all artifacts produced by the assignment pipeline."""

    subgoal_maps: np.ndarray
    has_global_goal: List[bool]
    global_goal_indices: List[int]
    global_goal_positions: List[Optional[Coord]]
    local_subgoal_positions: List[Optional[Coord]]
    global_paths: List[Optional[List[Coord]]]


def assign_and_plan(
    robot_positions: Sequence[Coord],
    goal_centers: Sequence[Coord],
    cost_matrix,
    global_map,
    fov_size: int,
    planner: Optional[AStarPlanner] = None,
) -> AssignmentResult:
    """
    Execute assignment + planning + local sub-goal extraction pipeline.
    """
    if fov_size % 2 == 0:
        raise ValueError("fov_size must be odd so that agents remain centered in FOV.")

    robots = [tuple(int(v) for v in pos) for pos in robot_positions]
    goals = [tuple(int(v) for v in pos) for pos in goal_centers]
    planner = planner or AStarPlanner()

    cost = _as_numpy(cost_matrix, len(robots), len(goals))
    grid = np.asarray(global_map)

    has_goal, goal_indices = _solve_assignment(cost, len(robots), len(goals))
    global_paths = _plan_paths(robots, goals, has_goal, goal_indices, grid, planner)
    subgoal_maps, local_subgoal_positions = _extract_local_subgoals(
        robots, global_paths, fov_size
    )

    global_goal_positions: List[Optional[Coord]] = []
    for idx, has in zip(goal_indices, has_goal):
        if has and 0 <= idx < len(goals):
            global_goal_positions.append(goals[idx])
        else:
            global_goal_positions.append(None)

    return AssignmentResult(
        subgoal_maps=subgoal_maps,
        has_global_goal=list(has_goal),
        global_goal_indices=list(goal_indices),
        global_goal_positions=global_goal_positions,
        local_subgoal_positions=local_subgoal_positions,
        global_paths=global_paths,
    )


def _as_numpy(cost_matrix, num_robots: int, num_goals: int) -> np.ndarray:
    if cost_matrix is None:
        return np.zeros((num_robots, num_goals), dtype=np.float32)

    if hasattr(cost_matrix, "detach"):
        cost_matrix = cost_matrix.detach().cpu().numpy()
    elif hasattr(cost_matrix, "cpu"):
        cost_matrix = cost_matrix.cpu().numpy()

    arr = np.asarray(cost_matrix, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("cost_matrix must be 2-D (N, M).")
    return arr


def _solve_assignment(cost: np.ndarray, num_robots: int, num_goals: int):
    if num_robots == 0:
        return np.zeros((0,), dtype=bool), np.zeros((0,), dtype=int)
    if num_goals == 0:
        return np.zeros((num_robots,), dtype=bool), -np.ones((num_robots,), dtype=int)

    row_ind, col_ind = linear_sum_assignment(cost)
    goal_indices = -np.ones((num_robots,), dtype=int)
    has_goal = np.zeros((num_robots,), dtype=bool)

    for row, col in zip(row_ind, col_ind):
        if row < num_robots and col < num_goals:
            goal_indices[row] = int(col)
            has_goal[row] = True

    return has_goal, goal_indices


def _plan_paths(
    robot_positions: Sequence[Coord],
    goal_centers: Sequence[Coord],
    has_goal: Sequence[bool],
    goal_indices: Sequence[int],
    global_map: np.ndarray,
    planner: AStarPlanner,
) -> List[Optional[List[Coord]]]:
    global_paths: List[Optional[List[Coord]]] = [None for _ in robot_positions]
    for ridx, has in enumerate(has_goal):
        if not has:
            continue
        goal_idx = goal_indices[ridx]
        if goal_idx < 0 or goal_idx >= len(goal_centers):
            continue
        start = robot_positions[ridx]
        goal = goal_centers[goal_idx]
        path = planner.plan(start, goal, global_map)
        global_paths[ridx] = path if path else None
    return global_paths


def _extract_local_subgoals(
    robot_positions: Sequence[Coord],
    global_paths: Sequence[Optional[List[Coord]]],
    fov_size: int,
):
    view_radius = fov_size // 2
    subgoal_channels: List[np.ndarray] = []
    local_targets: List[Optional[Coord]] = []

    for idx, path in enumerate(global_paths):
        channel = np.zeros((fov_size, fov_size), dtype=np.float32)
        local_target: Optional[Coord] = None
        if path:
            agent_pos = robot_positions[idx]
            for node in reversed(path):
                if _in_fov(agent_pos, node, view_radius):
                    local_target = node
                    row_offset = node[0] - agent_pos[0] + view_radius
                    col_offset = node[1] - agent_pos[1] + view_radius
                    if 0 <= row_offset < fov_size and 0 <= col_offset < fov_size:
                        channel[row_offset, col_offset] = 1.0
                    break
        subgoal_channels.append(channel)
        local_targets.append(local_target)

    stacked = np.stack(subgoal_channels, axis=0) if subgoal_channels else np.zeros(
        (0, fov_size, fov_size), dtype=np.float32
    )
    return stacked, local_targets


def _in_fov(center: Coord, point: Coord, radius: int) -> bool:
    return (
        abs(point[0] - center[0]) <= radius and abs(point[1] - center[1]) <= radius
    )
