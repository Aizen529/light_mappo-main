from typing import List, Optional, Sequence, Tuple

import numpy as np


class EnvCore(object):
    """
    Multi-robot coverage environment tailored for MAPPO training.

    The world is an H x W global occupancy grid with three semantic states per
    cell: uncovered (0), covered (1) and obstacle (blocked). Four heterogeneous
    robots (red, yellow, blue, green) navigate the grid to cover as many free
    cells as possible while respecting motion constraints and collision rules.
    """

    # Base action definitions used to compose agent-specific action sets.
    CARDINAL_ACTIONS: Tuple[Tuple[str, Tuple[int, int]], ...] = (
        ("stay", (0, 0)),
        ("up", (-1, 0)),
        ("down", (1, 0)),
        ("left", (0, -1)),
        ("right", (0, 1)),
    )
    EXTENDED_ACTIONS: Tuple[Tuple[str, Tuple[int, int]], ...] = CARDINAL_ACTIONS + (
        ("up_left", (-1, -1)),
        ("down_left", (1, -1)),
        ("up_right", (-1, 1)),
        ("down_right", (1, 1)),
    )

    DEFAULT_INITIAL_POSITIONS: Tuple[Tuple[int, int], ...] = (
        (1, 1),   # red
        (14, 7),  # yellow
        (7, 14),  # blue
        (18, 17), # green
    )

    COLOR_LEGEND = {
        "uncovered": (238, 240, 242),
        "covered": (200, 205, 210),
        "obstacle": (30, 30, 30),
        "red": (213, 52, 62),
        "yellow": (247, 189, 57),
        "blue": (54, 135, 194),
        "green": (80, 160, 90),
        "red_path": (247, 143, 152),
        "yellow_path": (252, 214, 138),
        "blue_path": (123, 178, 226),
        "green_path": (140, 199, 149),
    }

    def __init__(
        self,
        map_height: int = 20,
        map_width: int = 20,
        fov_size: int = 3,
        cover_reward: float = 1.0,
        max_episode_steps: Optional[int] = 500,
        obstacle_coords: Optional[Sequence[Tuple[int, int]]] = None,
        initial_positions: Optional[Sequence[Tuple[int, int]]] = None,
        seed: Optional[int] = None,
    ):
        if fov_size % 2 == 0:
            raise ValueError("fov_size must be an odd number to center the agent's view.")

        self.map_height = map_height
        self.map_width = map_width
        self.fov_size = fov_size
        self.view_radius = fov_size // 2
        self.cover_reward = cover_reward
        self.max_episode_steps = max_episode_steps or map_height * map_width * 2

        self._rng = np.random.default_rng(seed)

        # Agent metadata (red, yellow, blue share the same action set).
        self.agent_specs = [
            {"name": "red", "action_set": self.CARDINAL_ACTIONS},
            {"name": "yellow", "action_set": self.CARDINAL_ACTIONS},
            {"name": "blue", "action_set": self.CARDINAL_ACTIONS},
            {"name": "green", "action_set": self.EXTENDED_ACTIONS},
        ]
        self.agent_num = len(self.agent_specs)
        self.agent_action_sets = [tuple(spec["action_set"]) for spec in self.agent_specs]
        self.action_dims = [len(action_set) for action_set in self.agent_action_sets]
        # Retain compatibility with wrappers that expect a scalar attribute.
        self.action_dim = max(self.action_dims)

        self.obs_channels = 4  # history, neighbors, obstacles, coverage
        self.obs_dim = self.obs_channels * self.fov_size * self.fov_size

        self._preset_positions = (
            tuple(initial_positions) if initial_positions is not None else self.DEFAULT_INITIAL_POSITIONS
        )
        self._preset_consumed = False
        if self._preset_positions is not None:
            self._validate_coordinates(self._preset_positions)

        self.obstacle_coords = (
            tuple(obstacle_coords)
            if obstacle_coords is not None
            else self._build_default_obstacles(self._preset_positions)
        )
        self._validate_coordinates(self.obstacle_coords)

        self.obstacle_map = np.zeros((self.map_height, self.map_width), dtype=np.float32)
        for coord in self.obstacle_coords:
            self.obstacle_map[coord] = 1.0

        self.walkable_mask = self.obstacle_map == 0
        self.total_traversable_cells = int(self.walkable_mask.sum())
        if self.total_traversable_cells < self.agent_num:
            raise ValueError("Not enough traversable cells to place all agents.")

        self.coverage_map = np.zeros_like(self.obstacle_map)
        self.trail_owner = -np.ones((self.map_height, self.map_width), dtype=np.int32)
        self.history_maps: List[np.ndarray] = []
        self.agent_positions: List[Tuple[int, int]] = []
        self.coverage_count = 0
        self.current_step = 0
        self.episode_count = 0
        self.agent_colors = [
            np.array(self.COLOR_LEGEND[spec["name"]], dtype=np.uint8) for spec in self.agent_specs
        ]
        self.path_colors = self._build_path_colors()

        # Rendering control.
        self._render_scale = 36
        self._render_initialized = False
        self._render_artist = None
        self._render_fig = None
        self._render_ax = None
        self._render_text_artist = None

    # -------------------------------------------------------------------------
    # Public API expected by the MAPPO wrappers
    # -------------------------------------------------------------------------
    def reset(self) -> List[np.ndarray]:
        """
        Reset the episode, optionally using user-defined initial positions for
        the very first rollout, then sampling random non-overlapping spawn
        locations for subsequent episodes.
        """
        self.coverage_map = np.zeros_like(self.obstacle_map)
        self.trail_owner.fill(-1)
        self.history_maps = [np.zeros_like(self.obstacle_map) for _ in range(self.agent_num)]
        self.agent_positions = self._spawn_agents()
        self.coverage_count = 0
        self.current_step = 0
        self.episode_count += 1

        for agent_idx, pos in enumerate(self.agent_positions):
            self.history_maps[agent_idx][pos] = 1.0
            if self.coverage_map[pos] == 0:
                self.coverage_map[pos] = 1.0
                self.coverage_count += 1
            if self.trail_owner[pos] == -1:
                self.trail_owner[pos] = agent_idx

        return self._build_observations()

    def step(self, actions):
        """
        Execute a synchronous multi-agent step following the requested collision
        and motion-handling rules.
        """
        parsed_actions = self._normalize_actions(actions)

        raw_targets: List[Tuple[int, int]] = []
        action_labels: List[str] = []

        for agent_idx, action in enumerate(parsed_actions):
            action_idx, action_label, delta = self._decode_action(agent_idx, action)
            action_labels.append(action_label)
            current_row, current_col = self.agent_positions[agent_idx]
            target_row = current_row + delta[0]
            target_col = current_col + delta[1]
            raw_targets.append((target_row, target_col))

        projected_positions, attempted_moves = self._apply_environment_rules(raw_targets)
        rewards = []
        infos = []
        collisions = self._detect_collisions(projected_positions)
        final_positions = self._resolve_conflicts(projected_positions, collisions, attempted_moves)

        self.agent_positions = final_positions
        self.current_step += 1

        for agent_idx, pos in enumerate(self.agent_positions):
            row, col = pos
            newly_covered = 0.0
            if self.coverage_map[row, col] == 0:
                self.coverage_map[row, col] = 1.0
                self.coverage_count += 1
                newly_covered = self.cover_reward
                if self.trail_owner[row, col] == -1:
                    self.trail_owner[row, col] = agent_idx

            reward = newly_covered
            rewards.append(np.array([reward], dtype=np.float32))
            self.history_maps[agent_idx][row, col] = 1.0

            covered_ratio = self.coverage_count / float(self.total_traversable_cells)
            infos.append(
                {
                    "agent": self.agent_specs[agent_idx]["name"],
                    "action": action_labels[agent_idx],
                    "collision": collisions[agent_idx],
                    "made_progress": newly_covered > 0,
                    "position": pos,
                    "covered_ratio": covered_ratio,
                    "step": self.current_step,
                }
            )

        done = self._episode_done()
        dones = [done for _ in range(self.agent_num)]
        observations = self._build_observations()

        return [observations, rewards, dones, infos]

    def render(self, mode: str = "rgb_array"):
        """
        Render the global grid. In rgb_array mode an RGB numpy array is
        returned. In human mode matplotlib (when available) is used to visualize
        the board; otherwise a text-based fallback is printed.
        """
        frame = self._compose_frame()

        if mode == "rgb_array":
            return frame

        if mode != "human":
            raise NotImplementedError(f"Unsupported render mode: {mode}")

        try:
            import matplotlib.pyplot as plt  # type: ignore
        except ImportError:
            self._render_with_text()
            return frame

        if not self._render_initialized:
            plt.ion()
            self._render_fig, self._render_ax = plt.subplots(figsize=(4, 4))
            self._render_artist = self._render_ax.imshow(frame)
            self._render_ax.axis("off")
            self._render_text_artist = self._render_ax.text(
                0.02,
                1.02,
                "",
                transform=self._render_ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=12,
                color="black",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2},
            )
            self._render_initialized = True
        else:
            assert self._render_artist is not None
            self._render_artist.set_data(frame)

        if self._render_text_artist is not None:
            self._render_text_artist.set_text(self._format_step_annotation())

        assert self._render_fig is not None
        self._render_fig.canvas.draw()
        self._render_fig.canvas.flush_events()
        return frame

    def seed(self, seed: Optional[int]):
        """Reset the internal RNG with the provided seed."""
        self._rng = np.random.default_rng(seed)
        return seed

    def close(self):
        if self._render_fig is not None:
            try:
                import matplotlib.pyplot as plt  # type: ignore

                plt.close(self._render_fig)
            except ImportError:
                pass
        self._render_fig = None
        self._render_ax = None
        self._render_artist = None
        self._render_text_artist = None
        self._render_initialized = False

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _rectangle_coords(row_start: int, row_end: int, col_start: int, col_end: int) -> List[Tuple[int, int]]:
        """
        Generate inclusive grid coordinates for the rectangle defined by
        [row_start, row_end] x [col_start, col_end].
        """
        coords = []
        for r in range(row_start, row_end + 1):
            for c in range(col_start, col_end + 1):
                coords.append((r, c))
        return coords

    def _build_default_obstacles(self, reserved_coords: Optional[Sequence[Tuple[int, int]]]) -> Tuple[Tuple[int, int], ...]:
        """
        Construct the default obstacle layout from predefined rectangles and
        drop any cells that overlap reserved spawn coordinates.
        """
        rectangles = [
            (2, 12, 3, 6),
            (14, 16, 3, 5),
            (4, 10, 9, 12),
            (12, 15, 10, 11),
            (12, 15, 15, 16),
            (16, 18, 12, 14),
            (1, 2, 16, 17),
            (5, 9, 17, 18),
        ]

        coords = []
        for row_start, row_end, col_start, col_end in rectangles:
            coords.extend(self._rectangle_coords(row_start, row_end, col_start, col_end))

        reserved = set(reserved_coords or [])
        coords = [coord for coord in coords if coord not in reserved]
        # Deduplicate while keeping deterministic order.
        return tuple(sorted(set(coords)))

    def _validate_coordinates(self, coords: Sequence[Tuple[int, int]]):
        for row, col in coords:
            if not (0 <= row < self.map_height and 0 <= col < self.map_width):
                raise ValueError(f"Coordinate {(row, col)} exceeds the map boundary {self.map_height}x{self.map_width}.")

    def _spawn_agents(self) -> List[Tuple[int, int]]:
        if not self._preset_consumed and self._preset_positions is not None:
            candidates = list(self._preset_positions)
            self._preset_consumed = True
        else:
            candidates = []
            free_cells = np.argwhere(self.walkable_mask)
            if len(free_cells) < self.agent_num:
                raise RuntimeError("Insufficient free cells for spawning agents.")
            perm = self._rng.permutation(len(free_cells))
            for idx in perm:
                row, col = map(int, free_cells[idx])
                coord = (row, col)
                if coord in candidates:
                    continue
                candidates.append(coord)
                if len(candidates) == self.agent_num:
                    break

        for coord in candidates:
            if not self.walkable_mask[coord]:
                raise ValueError(f"Spawn coordinate {coord} collides with an obstacle.")
        if len(set(candidates)) != self.agent_num:
            raise ValueError("Spawn coordinates must be unique per agent.")

        return candidates

    def _normalize_actions(self, actions):
        if actions is None:
            return [None for _ in range(self.agent_num)]

        if isinstance(actions, np.ndarray):
            if actions.ndim == 1 and self.agent_num == 1:
                return [actions]
            if actions.shape[0] != self.agent_num:
                raise ValueError(f"Expected {self.agent_num} actions, received {actions.shape[0]}.")
            return [actions[idx] for idx in range(self.agent_num)]
        if isinstance(actions, Sequence):
            if len(actions) != self.agent_num:
                raise ValueError(f"Expected {self.agent_num} actions, received {len(actions)}.")
            return list(actions)

        raise ValueError("Unsupported action format supplied to the environment.")

    def _build_path_colors(self) -> List[np.ndarray]:
        colors = []
        for spec in self.agent_specs:
            key = f"{spec['name']}_path"
            base = self.COLOR_LEGEND.get(key, self.COLOR_LEGEND.get(spec["name"], (255, 255, 255)))
            colors.append(np.array(base, dtype=np.uint8))
        return colors

    def _decode_action(self, agent_idx: int, action) -> Tuple[int, str, Tuple[int, int]]:
        action_set = self.agent_action_sets[agent_idx]
        if action is None:
            action_idx = int(self._rng.integers(0, len(action_set)))
        else:
            vector = np.asarray(action)
            if vector.ndim == 0:
                action_idx = int(vector)
            else:
                action_idx = int(np.argmax(vector))
        action_idx = int(np.clip(action_idx, 0, len(action_set) - 1))
        action_name, delta = action_set[action_idx]
        return action_idx, action_name, delta

    def _apply_environment_rules(self, raw_targets: List[Tuple[int, int]]):
        projected = []
        attempted_moves = []
        for agent_idx, target in enumerate(raw_targets):
            row, col = target
            current = self.agent_positions[agent_idx]
            attempted = current != target
            valid = (
                0 <= row < self.map_height
                and 0 <= col < self.map_width
                and self.obstacle_map[row, col] == 0
            )
            if valid:
                projected.append(target)
                attempted_moves.append(attempted)
            else:
                projected.append(current)
                attempted_moves.append(False)
        return projected, attempted_moves

    def _detect_collisions(self, projected_positions: List[Tuple[int, int]]) -> List[bool]:
        collisions = [False for _ in range(self.agent_num)]

        # Same target collisions.
        position_to_agents = {}
        for idx, pos in enumerate(projected_positions):
            position_to_agents.setdefault(pos, []).append(idx)

        for agents in position_to_agents.values():
            if len(agents) > 1:
                for agent_idx in agents:
                    collisions[agent_idx] = True

        # Swap collisions (two agents attempting to exchange positions).
        for i in range(self.agent_num):
            for j in range(i + 1, self.agent_num):
                if (
                    projected_positions[i] == self.agent_positions[j]
                    and projected_positions[j] == self.agent_positions[i]
                    and projected_positions[i] != projected_positions[j]
                ):
                    collisions[i] = True
                    collisions[j] = True

        return collisions

    def _resolve_conflicts(
        self,
        projected_positions: List[Tuple[int, int]],
        collisions: List[bool],
        attempted_moves: List[bool],
    ) -> List[Tuple[int, int]]:
        resolved = list(projected_positions)
        # Agents that collided stay put.
        for idx, collided in enumerate(collisions):
            if collided:
                resolved[idx] = self.agent_positions[idx]

        # Swap collisions supersede attempted moves.
        for i in range(self.agent_num):
            for j in range(i + 1, self.agent_num):
                if (
                    projected_positions[i] == self.agent_positions[j]
                    and projected_positions[j] == self.agent_positions[i]
                    and attempted_moves[i]
                    and attempted_moves[j]
                ):
                    resolved[i] = self.agent_positions[i]
                    resolved[j] = self.agent_positions[j]
        return resolved

    def _build_observations(self) -> List[np.ndarray]:
        observations: List[np.ndarray] = []
        for agent_idx in range(self.agent_num):
            center = self.agent_positions[agent_idx]
            history = self._extract_patch(self.history_maps[agent_idx], center)
            neighbors = self._build_neighbor_channel(agent_idx)
            obstacles = self._extract_patch(self.obstacle_map, center)
            covered = self._extract_patch(self.coverage_map, center)
            stacked = np.stack([history, neighbors, obstacles, covered], axis=0)
            observations.append(stacked.reshape(-1).astype(np.float32))
        return observations

    def _extract_patch(self, field: np.ndarray, center: Tuple[int, int]) -> np.ndarray:
        patch = np.zeros((self.fov_size, self.fov_size), dtype=np.float32)
        center_row, center_col = center

        row_start = center_row - self.view_radius
        row_end = center_row + self.view_radius + 1
        col_start = center_col - self.view_radius
        col_end = center_col + self.view_radius + 1

        src_row_start = max(0, row_start)
        src_row_end = min(self.map_height, row_end)
        src_col_start = max(0, col_start)
        src_col_end = min(self.map_width, col_end)

        dest_row_start = src_row_start - row_start
        dest_row_end = dest_row_start + (src_row_end - src_row_start)
        dest_col_start = src_col_start - col_start
        dest_col_end = dest_col_start + (src_col_end - src_col_start)

        patch[dest_row_start:dest_row_end, dest_col_start:dest_col_end] = field[
            src_row_start:src_row_end, src_col_start:src_col_end
        ]
        return patch

    def _build_neighbor_channel(self, agent_idx: int) -> np.ndarray:
        channel = np.zeros((self.fov_size, self.fov_size), dtype=np.float32)
        center_row, center_col = self.agent_positions[agent_idx]
        for other_idx, pos in enumerate(self.agent_positions):
            if other_idx == agent_idx:
                continue
            row_offset = pos[0] - center_row
            col_offset = pos[1] - center_col
            if abs(row_offset) <= self.view_radius and abs(col_offset) <= self.view_radius:
                channel[row_offset + self.view_radius, col_offset + self.view_radius] = 1.0
        return channel

    def _episode_done(self) -> bool:
        all_covered = self.coverage_count >= self.total_traversable_cells
        reached_limit = self.current_step >= self.max_episode_steps
        return all_covered or reached_limit

    def _compose_frame(self) -> np.ndarray:
        frame = np.zeros((self.map_height, self.map_width, 3), dtype=np.uint8)
        frame[:, :] = self.COLOR_LEGEND["uncovered"]

        covered_mask = self.coverage_map == 1
        frame[covered_mask] = self.COLOR_LEGEND["covered"]

        for agent_idx in range(self.agent_num):
            path_mask = self.trail_owner == agent_idx
            frame[path_mask] = self.path_colors[agent_idx]

        obstacle_mask = self.obstacle_map == 1
        frame[obstacle_mask] = self.COLOR_LEGEND["obstacle"]

        scale = self._render_scale
        frame = np.kron(frame, np.ones((scale, scale, 1), dtype=np.uint8))
        frame = self._draw_grid_lines(frame, scale)
        frame = self._draw_agent_circles(frame, scale)
        return frame

    def _draw_grid_lines(self, frame: np.ndarray, scale: int) -> np.ndarray:
        """Draw subtle grid lines between upsampled cells for visualization clarity."""
        color = np.array((60, 60, 60), dtype=np.uint8)
        height, width, _ = frame.shape
        for row in range(scale, height, scale):
            frame[row - 1 : row + 1, :, :] = color
        for col in range(scale, width, scale):
            frame[:, col - 1 : col + 1, :] = color
        return frame

    def _draw_agent_circles(self, frame: np.ndarray, scale: int) -> np.ndarray:
        """Render agents as colored disks centered in their respective cells."""
        for agent_idx, (row, col) in enumerate(self.agent_positions):
            center_r = row * scale + scale // 2
            center_c = col * scale + scale // 2
            radius = max(scale // 2 - 2, 2)

            r_min = max(center_r - radius, 0)
            r_max = min(center_r + radius, frame.shape[0] - 1)
            c_min = max(center_c - radius, 0)
            c_max = min(center_c + radius, frame.shape[1] - 1)

            region = frame[r_min : r_max + 1, c_min : c_max + 1]
            rows = np.arange(r_min, r_max + 1)[:, None]
            cols = np.arange(c_min, c_max + 1)[None, :]
            mask = (rows - center_r) ** 2 + (cols - center_c) ** 2 <= radius ** 2
            region[mask] = self.agent_colors[agent_idx]

        return frame

    def _format_step_annotation(self) -> str:
        """Readable step counter displayed above the grid."""
        return f"Steps: {self.current_step}"

    def _render_with_text(self):
        legend = {
            0: ".",
            1: "#",
            2: "R",
            3: "Y",
            4: "B",
            5: "G",
        }
        board = np.zeros((self.map_height, self.map_width), dtype=int)
        board[self.coverage_map == 1] = 1
        board[self.obstacle_map == 1] = 8
        for idx, (row, col) in enumerate(self.agent_positions):
            board[row, col] = 2 + idx

        text_rows = []
        for r in range(self.map_height):
            row_chars = []
            for c in range(self.map_width):
                value = board[r, c]
                if value == 8:
                    row_chars.append("X")
                else:
                    row_chars.append(legend.get(value, "."))
            text_rows.append(" ".join(row_chars))
        print("\n".join(text_rows))
