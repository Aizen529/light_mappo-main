from typing import Sequence

import torch
from torch import nn

from .subgoal import Subgoal


class GoalGeomEncoder(nn.Module):
    """Encode geometric attributes of subgoals into embeddings."""

    def __init__(
        self,
        map_height: int,
        map_width: int,
        d_model: int = 64,
        hidden_dim: int = 128,
    ):
        super().__init__()
        if map_height <= 0 or map_width <= 0:
            raise ValueError("map dimensions must be positive.")

        self.map_height = float(map_height)
        self.map_width = float(map_width)
        self.total_cells = self.map_height * self.map_width

        in_dim = 8  # centroid(2) + area raw/norm + bbox(w,h raw/norm)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model),
        )

    def _build_feature_tensor(self, subgoals: Sequence[Subgoal]) -> torch.Tensor:
        if len(subgoals) == 0:
            return torch.zeros((0, 8), dtype=torch.float32)

        feats = []
        for sg in subgoals:
            cy, cx = sg.centroid
            area = float(sg.area)
            min_row, min_col, max_row, max_col = sg.bbox
            width = float(max_col - min_col + 1)
            height = float(max_row - min_row + 1)

            cx_norm = max(min(cx / self.map_width, 1.0), 0.0)
            cy_norm = max(min(cy / self.map_height, 1.0), 0.0)
            area_norm = area / self.total_cells
            width_norm = width / self.map_width
            height_norm = height / self.map_height

            feats.append(
                [
                    cx_norm,
                    cy_norm,
                    area,
                    area_norm,
                    width,
                    height,
                    width_norm,
                    height_norm,
                ]
            )

        return torch.tensor(feats, dtype=torch.float32)

    def forward(self, subgoals: Sequence[Subgoal]) -> torch.Tensor:
        """
        Args:
            subgoals: sequence of Subgoal objects.
        Returns:
            torch.Tensor: shape (M, d_model) embeddings, M=len(subgoals).
        """
        feat_tensor = self._build_feature_tensor(subgoals)
        return self.mlp(feat_tensor)
