from __future__ import annotations

import argparse
import math
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Inspect the triple pendulum mechanism with an open-loop motor command.")
parser.add_argument("--task", type=str, default="TriplePendulum-SwingUpBalance-Direct-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=900)
parser.add_argument("--amplitude", type=float, default=0.8)
parser.add_argument("--frequency", type=float, default=0.35)
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=300)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]]
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import triple_pendulum_isaac  # noqa: F401
from triple_pendulum_isaac.tasks.direct.triple_inverted_pendulum.env import (
    TripleInvertedPendulumBalanceEnvCfg,
    TripleInvertedPendulumSwingUpBalanceEnvCfg,
    TripleInvertedPendulumSwingUpEnvCfg,
)


CFG_BY_TASK = {
    "TriplePendulum-Balance-Direct-v0": TripleInvertedPendulumBalanceEnvCfg,
    "TriplePendulum-SwingUp-Direct-v0": TripleInvertedPendulumSwingUpEnvCfg,
    "TriplePendulum-SwingUpBalance-Direct-v0": TripleInvertedPendulumSwingUpBalanceEnvCfg,
}


def main() -> None:
    if args_cli.task not in CFG_BY_TASK:
        raise ValueError(f"Unsupported task '{args_cli.task}'. Expected one of: {sorted(CFG_BY_TASK)}")

    env_cfg = CFG_BY_TASK[args_cli.task]()
    env_cfg.scene.num_envs = args_cli.num_envs

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=f"logs/mechanism/{args_cli.task}",
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )

    obs = env.reset()
    dt = env.unwrapped.step_dt
    device = env.unwrapped.device

    for step in range(args_cli.steps):
        phase = 2.0 * math.pi * args_cli.frequency * step * dt
        cmd = args_cli.amplitude * math.sin(phase)
        actions = torch.full((env.unwrapped.num_envs, 1), cmd, device=device)
        obs, _, terminated, truncated, _ = env.step(actions)
        if torch.any(terminated) or torch.any(truncated):
            obs = env.reset()
        if not simulation_app.is_running():
            break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
