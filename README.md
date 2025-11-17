# light_mappo

Lightweight version of MAPPO to help you quickly migrate to your local environment.

- [Video (in Chinese)](https://www.bilibili.com/video/BV1bd4y1L73N)  
This is a translated English version. Please click [here](README_CN.md) for the orginal Chinese readme.

This code has been used in the following paper:

```bash
@inproceedings{he2024intelligent,
  title={Intelligent Decentralized Multiple Access via Multi-Agent Deep Reinforcement Learning},
  author={He, Yuxuan and Gang, Xinyuan and Gao, Yayu},
  booktitle={2024 IEEE Wireless Communications and Networking Conference (WCNC)},
  pages={1--6},
  year={2024},
  organization={IEEE}
}
@article{qiu2024enhancing,
  title={Enhancing UAV Communications in Disasters: Integrating ESFM and MAPPO for Superior Performance},
  author={Qiu, Wen and Shao, Xun and Loke, Seng W and He, Zhiqiang and Alqahtani, Fayez and Masui, Hiroshi},
  journal={Journal of Circuits, Systems and Computers},
  year={2024},
  publisher={World Scientific}
}
@article{qiu2024optimizing,
  title={Optimizing Drone Energy Use for Emergency Communications in Disasters via Deep Reinforcement Learning},
  author={Qiu, Wen and Shao, Xun and Masui, Hiroshi and Liu, William},
  journal={Future Internet},
  volume={16},
  number={7},
  pages={245},
  year={2024},
  publisher={MDPI}
}
@inproceedings{yu2024path,
  title={Path Planning for Multi-AGV Systems Based on Globally Guided Reinforcement Learning Approach},
  author={Yu, Lanlin and Wang, Yusheng and Sheng, Zixiang and Xu, Pengfei and He, Zhiqiang and Du, Haibo},
  booktitle={2024 IEEE International Conference on Unmanned Systems (ICUS)},
  pages={819--825},
  year={2024},
  organization={IEEE}
}
```

## Table of Contents

- [Background](#Background)
- [Installation](#Installation)
- [Usage](#Usage)

## Background

The original MAPPO code was too complex in terms of environment encapsulation, so this project directly extracts and encapsulates the environment. This makes it easier to transfer the MAPPO code to your own project.

## Installation

Simply download the code, create a Conda environment, and then run the code, adding packages as needed. Specific packages will be added later.

## Usage

`envs/env_core.py` now contains a fully functional grid-coverage environment that matches the problem statement:

- **Map:** configurable H×W global occupancy grid (defaults to 8×8) with uncovered, covered, and obstacle states. `DEFAULT_OBSTACLES` controls the static obstacle layout.
- **Agents:** four robots (red, yellow, blue, green). Red/yellow/blue share five cardinal actions, while the green robot can additionally move diagonally (nine discrete actions total).
- **Observations:** every step each agent receives a 3×3 local map with four semantic channels (self trajectory history inside the FOV, neighbors’ relative positions, obstacles, and already-covered cells). The observation returned to MAPPO is the flattened tensor so it plugs directly into the existing networks.
- **Dynamics and collisions:** movement happens synchronously. Invalid moves (off-map or into obstacles) are turned into `stay`, multiple agents trying to enter the same cell all stay in place, and head-on swaps are cancelled for both participants.
- **Reward:** +1 when an agent covers a previously unseen traversable cell. The episode terminates once every free cell is covered or when the max episode length is reached.
- **Reset logic:** the first episode spawns robots at `DEFAULT_INITIAL_POSITIONS`; subsequent episodes sample random non-overlapping spawn cells that avoid obstacles. You can override these defaults by editing the constants near the top of `EnvCore` or by instantiating `EnvCore` with custom parameters in `make_train_env`.
- **Visualization:** `EnvCore.render()` returns an RGB frame (and optionally displays it with matplotlib); there is also an ASCII fallback when running in a pure terminal.

The MAPPO wrappers in `envs/env_discrete.py` automatically pick up the heterogeneous action spaces, so no extra glue code is needed. `train/train.py` now instantiates the discrete environment by default and infers the agent count from the environment, so the standard training entrypoint works out of the box:

```bash
python train/train.py --algorithm_name rmappo --experiment_name grid_demo --num_env_steps 200000
```

If you want to preview the environment without training, run the visualization helper:

```bash
python scripts/demo_grid_env.py --episodes 5 --sleep 0.1
```

This script samples random actions, calls the new renderer, and prints per-step coverage statistics. Use `--render_mode rgb_array` to consume frames in downstream tooling.

## Cite this work

If you use `light_mappo`, please cite:

```bibtex
@software{light_mappo,
  author  = {Zhiqiang He},
  title   = {light\_mappo: Lightweight MAPPO implementation},
  year    = {2025},
  url     = {https://github.com/tinyzqh/light_mappo},
  note    = {Version v0.1.0}
}
```

## Related Efforts

- [on-policy](https://github.com/marlbenchmark/on-policy) - 💌 Learn the author implementation of MAPPO.

## Maintainers

[@tinyzqh](https://github.com/tinyzqh).

## Translator
[@tianyu-z](https://github.com/tianyu-z)

## License

[MIT](LICENSE) © tinyzqh
