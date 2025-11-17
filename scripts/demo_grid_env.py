import argparse
import time
from typing import List

import numpy as np

from envs.env_discrete import DiscreteActionEnv


def _format_rewards(rewards: np.ndarray) -> List[float]:
    return [float(r[0]) for r in rewards]


def run_demo(args):
    env = DiscreteActionEnv()
    env.env.seed(args.seed)
    for episode in range(args.episodes):
        obs = env.reset()
        if args.verbose:
            print(f"Episode {episode} reset. Obs shape: {obs.shape}")
        for step in range(args.episode_length):
            actions = [space.sample() for space in env.action_space]
            obs, rewards, dones, infos = env.step(actions)
            frame = env.env.render(mode=args.render_mode)
            if args.verbose:
                covered_ratio = infos[0]["covered_ratio"]
                print(
                    f"Episode {episode} Step {step} rewards {_format_rewards(rewards)} "
                    f"covered={covered_ratio:.2f}"
                )
            if args.render_mode == "rgb_array" and args.verbose:
                print(f"Frame shape: {frame.shape}")
            time.sleep(args.sleep)
            if all(dones):
                break
    env.env.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Demo script to visualize the grid coverage environment.")
    parser.add_argument("--episodes", type=int, default=3, help="Number of demo episodes.")
    parser.add_argument("--episode_length", type=int, default=50, help="Max steps per episode.")
    parser.add_argument("--sleep", type=float, default=0.2, help="Delay between renders (seconds).")
    parser.add_argument(
        "--render_mode",
        type=str,
        default="human",
        choices=["human", "rgb_array"],
        help="Render mode passed to EnvCore.render.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for the environment.")
    parser.add_argument("--verbose", action="store_true", help="Print debug info during the rollout.")
    return parser.parse_args()


if __name__ == "__main__":
    run_demo(parse_args())
