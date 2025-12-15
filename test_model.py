#!/usr/bin/env python
"""
Evaluation script for MAPPO checkpoints with rendering and metric export.

Features:
- Deterministic (argmax) action selection per agent.
- MP4 video per episode (frames from EnvCore.render(rgb_array)).
- Metrics per episode + mean row saved to CSV; meta saved to JSON.
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch

from algorithms.algorithm.rMAPPOPolicy import RMAPPOPolicy as Policy
from config import get_config
from envs.env_discrete import DiscreteActionEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained MAPPO models with rendering.")
    parser.add_argument(
        "--run_dir",
        type=str,
        default="results/MyEnv/MyEnv/mappo/check/run1",
        help="Path to run directory containing models subfolder.",
    )
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes.")
    parser.add_argument(
        "--max_episode_steps",
        type=int,
        default=None,
        help="Override EnvCore max_episode_steps. None=use env default.",
    )
    parser.add_argument("--fps", type=int, default=10, help="FPS for saved videos.")
    parser.add_argument(
        "--render_first_n",
        type=int,
        default=0,
        help="Show interactive window for first N episodes (still records all).",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="test_results",
        help="Root folder to store test outputs (auto-increment testX).",
    )
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda:<id>. Defaults to auto.")
    parser.add_argument("--seed", type=int, default=1, help="Seed for env/torch/numpy.")
    parser.add_argument(
        "--subgoal_update_interval",
        type=int,
        default=1,
        help="Forwarded to EnvCore via DiscreteActionEnv.",
    )
    parser.add_argument(
        "--guidance_reward",
        type=float,
        default=0.2,
        help="Forwarded to EnvCore via DiscreteActionEnv.",
    )
    return parser.parse_args()


def make_eval_args():
    """Create a minimal args namespace for Policy compatible with training defaults."""
    base_parser = get_config()
    base_args = base_parser.parse_known_args([])[0]
    base_args.use_naive_recurrent_policy = False
    base_args.use_recurrent_policy = False
    base_args.model_dir = None
    return base_args


def allocate_output_dir(root: Path, prefix: str = "test") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    existing = []
    for path in root.iterdir():
        if path.is_dir() and path.name.startswith(prefix):
            suffix = path.name[len(prefix) :]
            if suffix.isdigit():
                existing.append(int(suffix))
    next_idx = max(existing) + 1 if existing else 1
    out_dir = root / f"{prefix}{next_idx}"
    (out_dir / "videos").mkdir(parents=True, exist_ok=True)
    return out_dir


def init_rnn_states(num_agents: int, hidden_size: int, recurrent_n: int) -> List[np.ndarray]:
    return [np.zeros((1, recurrent_n, hidden_size), dtype=np.float32) for _ in range(num_agents)]


def init_masks(num_agents: int) -> List[np.ndarray]:
    return [np.ones((1, 1), dtype=np.float32) for _ in range(num_agents)]


def write_video_frame(writer, frame_rgb: np.ndarray):
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    writer.write(frame_bgr)


def main():
    args = parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(1)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    models_dir = Path(args.run_dir) / "models"
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    output_dir = allocate_output_dir(Path(args.output_root))
    videos_dir = output_dir / "videos"
    metrics_path = output_dir / "metrics.csv"
    meta_path = output_dir / "meta.json"

    env = DiscreteActionEnv(
        train_mode=False,
        max_episode_steps=args.max_episode_steps,
        subgoal_update_interval=args.subgoal_update_interval,
        guidance_reward=args.guidance_reward,
    )
    env.seed(args.seed)
    base_env = env.env
    num_agents = env.num_agent

    policy_args = make_eval_args()
    policy_args.device = device
    policies = []
    for agent_id in range(num_agents):
        policy = Policy(
            policy_args,
            env.observation_space[agent_id],
            env.share_observation_space[agent_id],
            env.action_space[agent_id],
            device=device,
        )
        actor_path = models_dir / f"actor_agent{agent_id}.pt"
        if not actor_path.exists():
            raise FileNotFoundError(f"Missing actor checkpoint: {actor_path}")
        state_dict = torch.load(actor_path, map_location=device)
        policy.actor.load_state_dict(state_dict)
        policy.actor.eval()
        policies.append(policy)

    episode_rows = []
    success_count = 0

    for ep in range(args.episodes):
        obs = env.reset()
        rnn_states = init_rnn_states(num_agents, policy_args.hidden_size, policy_args.recurrent_N)
        masks = init_masks(num_agents)
        visit_counts = np.zeros_like(base_env.obstacle_map, dtype=np.int32)
        for pos in base_env.agent_positions:
            if base_env.walkable_mask[pos]:
                visit_counts[pos] += 1

        time_to_95 = None
        first_frame = base_env.render(mode="rgb_array")
        video_path = videos_dir / f"ep{ep:03d}.mp4"
        height, width = first_frame.shape[0], first_frame.shape[1]
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            args.fps,
            (width, height),
        )
        write_video_frame(writer, first_frame)
        if ep < args.render_first_n:
            base_env.render(mode="human")

        done = False
        while not done:
            actions: List[int] = []
            for agent_id in range(num_agents):
                obs_i = obs[agent_id].reshape(1, -1)
                act, rnn_states[agent_id] = policies[agent_id].act(
                    obs_i, rnn_states[agent_id], masks[agent_id], deterministic=True
                )
                act_np = act.squeeze().detach().cpu().numpy()
                actions.append(int(act_np))

            obs, rewards, dones, infos = env.step(np.array(actions))
            frame = base_env.render(mode="rgb_array")
            write_video_frame(writer, frame)
            if ep < args.render_first_n:
                base_env.render(mode="human")

            covered_ratio = infos[0]["covered_ratio"]
            if time_to_95 is None and covered_ratio >= 0.95:
                time_to_95 = base_env.current_step

            for pos in base_env.agent_positions:
                if base_env.walkable_mask[pos]:
                    visit_counts[pos] += 1

            done = bool(dones[0])

        writer.release()

        cov_pct = 100.0 * base_env.coverage_count / float(base_env.total_traversable_cells)
        dup_cells = int(((visit_counts > 1) & base_env.walkable_mask).sum())
        dup_cov_pct = 100.0 * dup_cells / float(base_env.total_traversable_cells)
        time_step = int(time_to_95) if time_to_95 is not None else -1
        success_flag = 1 if time_step != -1 else 0
        success_count += success_flag

        episode_rows.append(
            {
                "episode": ep,
                "cov_pct": cov_pct,
                "dup_cov_pct": dup_cov_pct,
                "time_step": time_step,
                "steps": base_env.current_step,
                "success_95": success_flag,
            }
        )

    cov_mean = float(np.mean([row["cov_pct"] for row in episode_rows])) if episode_rows else 0.0
    dup_mean = float(np.mean([row["dup_cov_pct"] for row in episode_rows])) if episode_rows else 0.0
    steps_mean = float(np.mean([row["steps"] for row in episode_rows])) if episode_rows else 0.0
    time_success = [row["time_step"] for row in episode_rows if row["time_step"] != -1]
    time_mean_success_only = float(np.mean(time_success)) if time_success else -1.0
    success_rate = success_count / float(len(episode_rows)) if episode_rows else 0.0

    episode_rows.append(
        {
            "episode": "mean",
            "cov_pct": cov_mean,
            "dup_cov_pct": dup_mean,
            "time_step": time_mean_success_only,
            "steps": steps_mean,
            "success_95": success_rate,
        }
    )

    with metrics_path.open("w", newline="") as f:
        fieldnames = ["episode", "cov_pct", "dup_cov_pct", "time_step", "steps", "success_95"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(episode_rows)

    meta = {
        "models_dir": str(models_dir.resolve()),
        "config_path": str(Path("config.py").resolve()),
        "run_dir": str(Path(args.run_dir).resolve()),
        "episodes": args.episodes,
        "seed": args.seed,
        "max_episode_steps": args.max_episode_steps,
        "fps": args.fps,
        "render_first_n": args.render_first_n,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "output_dir": str(output_dir.resolve()),
    }
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)

    env.close()


if __name__ == "__main__":
    main()
