import math
from typing import Sequence, Union

import torch
from torch import nn


TensorLike = Union[torch.Tensor, Sequence[Sequence[float]]]


class RobotGeomEncoder(nn.Module):
    """Encode per-robot geometric information into a fixed-size embedding."""

    def __init__(
        self,
        map_height: int,
        map_width: int,
        d_model: int = 64,
        id_embed_dim: int = 4,
        hidden_dim: int = 128,
    ):
        super().__init__()
        if map_height <= 0 or map_width <= 0:
            raise ValueError("map dimensions must be positive.")

        self.map_height = float(map_height)
        self.map_width = float(map_width)
        self.d_model = d_model

        self.id_embedding = nn.Embedding(2, id_embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(2 + id_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model),
        )

    @torch.no_grad()
    def _convert_positions(self, positions: TensorLike) -> torch.Tensor:
        if isinstance(positions, torch.Tensor):
            return positions.to(dtype=torch.float32)

        return torch.as_tensor(positions, dtype=torch.float32)

    @torch.no_grad()
    def _convert_ids(self, robot_ids: Union[torch.Tensor, Sequence[int]]) -> torch.Tensor:
        if isinstance(robot_ids, torch.Tensor):
            return robot_ids.to(dtype=torch.long)

        return torch.as_tensor(robot_ids, dtype=torch.long)

    def forward(
        self,
        positions: TensorLike,
        robot_ids: Union[torch.Tensor, Sequence[int]],
    ) -> torch.Tensor:
        """
        Args:
            positions: shape (N, 2) positions in (row, col) order.
            robot_ids: shape (N,) integer ids (0: default robots, 1: green robot).
        Returns:
            torch.Tensor: shape (N, d_model) of per-robot embeddings.
        """
        pos_tensor = self._convert_positions(positions)
        if pos_tensor.ndim != 2 or pos_tensor.shape[-1] != 2:
            raise ValueError("positions must have shape (N, 2).")

        robot_ids_tensor = self._convert_ids(robot_ids)
        if robot_ids_tensor.shape[0] != pos_tensor.shape[0]:
            raise ValueError("robot_ids length must match number of positions.")

        x_norm = torch.clamp(pos_tensor[:, 1] / self.map_width, min=0.0, max=1.0)
        y_norm = torch.clamp(pos_tensor[:, 0] / self.map_height, min=0.0, max=1.0)
        geom_feats = torch.stack([x_norm, y_norm], dim=-1)

        id_embs = self.id_embedding(robot_ids_tensor)
        feats = torch.cat([geom_feats, id_embs], dim=-1)
        return self.mlp(feats)
