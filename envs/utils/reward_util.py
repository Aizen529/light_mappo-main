from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple


def compute_rewards(
    newly_covered: Sequence[bool],
    has_subgoal: Sequence[bool],
    prev_goal_dists: Sequence[Optional[float]],
    curr_goal_dists: Sequence[Optional[float]],
    *,
    train_mode: bool,
    coverage_reward: float = 1.0,
    guidance_reward: float = 0.2,
) -> Tuple[List[float], float]:
    """
    Compute per-agent rewards and their global sum.
    """
    if not (
        len(newly_covered)
        == len(has_subgoal)
        == len(prev_goal_dists)
        == len(curr_goal_dists)
    ):
        raise ValueError("All reward inputs must share the same length.")

    local_rewards: List[float] = []

    for idx, covered in enumerate(newly_covered):
        reward = coverage_reward if covered else 0.0
        if train_mode and has_subgoal[idx]:
            prev_dist = prev_goal_dists[idx]
            curr_dist = curr_goal_dists[idx]
            if prev_dist is not None and curr_dist is not None:
                delta = prev_dist - curr_dist
                if delta > 0:
                    reward += guidance_reward
                elif delta < 0:
                    reward -= guidance_reward

        local_rewards.append(reward)

    global_reward = float(sum(local_rewards))
    return local_rewards, global_reward
