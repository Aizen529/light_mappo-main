import math
from typing import Optional

import torch
from torch import nn


class GeomCostNet(nn.Module):
    """Compute robot-to-subgoal cost matrix using scaled dot-product attention."""

    def __init__(
        self,
        d_model: int = 64,
        attn_dim: Optional[int] = None,
    ):
        super().__init__()
        attn_dim = attn_dim or d_model
        self.scale = math.sqrt(attn_dim)
        self.query_proj = nn.Linear(d_model, attn_dim, bias=False)
        self.key_proj = nn.Linear(d_model, attn_dim, bias=False)

    def forward(self, robot_embs: torch.Tensor, goal_embs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            robot_embs: tensor of shape (N, d_model).
            goal_embs: tensor of shape (M, d_model).
        Returns:
            torch.Tensor: cost matrix of shape (N, M) ready for Hungarian solver.
        """
        if robot_embs.ndim != 2:
            raise ValueError("robot_embs must be 2D (N, d_model).")
        if goal_embs.ndim != 2:
            raise ValueError("goal_embs must be 2D (M, d_model).")

        N = robot_embs.shape[0]
        M = goal_embs.shape[0]
        if N == 0 or M == 0:
            return torch.zeros((N, M), dtype=robot_embs.dtype, device=robot_embs.device)

        Q = self.query_proj(robot_embs)
        K = self.key_proj(goal_embs)
        logits = (Q @ K.T) / self.scale
        # Softmax over goals for each robot (row-wise).
        sim = torch.softmax(logits, dim=1)
        return -sim
