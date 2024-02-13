# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to an environment with random action agent."""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.orbit.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Orbit environments.")
parser.add_argument(
    "--cpu", action="store_true", default=False, help="Use CPU pipeline."
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--motion_primitive",
    type=str,
    default=None,
    help=(
        "Wether to use a motion primitive for the training. The supported ones depend in the environment: ProDMP,"
        " etc..."
    ),
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import matplotlib.pyplot as plt
import torch
import traceback

import carb

import omni.isaac.contrib_tasks  # noqa: F401
import omni.isaac.orbit_tasks  # noqa: F401
from omni.isaac.orbit_tasks.utils import parse_env_cfg


def main():
    """Random actions agent with Orbit environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs
    )
    # create environment
    task_name = args_cli.task
    if args_cli.motion_primitive is not None:
        task_name = "gym_" + args_cli.motion_primitive + "/" + task_name
    env = gym.make(task_name, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()
    # simulate environment
    rewards = []
    infos = []
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # sample actions from -1 to 1
            actions = (
                2
                * torch.rand(
                    (args_cli.num_envs, env.action_space.shape[0]),
                    device=env.unwrapped.device,
                )
                - 1
            )
            # actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
            # apply actions
            _, reward, _, _, info = env.step(actions)
            rewards.append(reward[0].cpu().numpy())
            infos.append(info)
    # close the simulator
    # plot_trajectories(rewards, infos)
    env.close()


def plot_trajectories(rewards, infos):
    _, axs = plt.subplots(2, 4)
    for i in range(len(rewards)):
        position = infos[i]["positions"][0].transpose(0, 1).cpu().tolist()
        axs[0, 0].plot(position[0], "b")
        axs[0, 1].plot(position[1], "b")
        axs[0, 2].plot(position[2], "b")
        axs[0, 3].plot(position[3], "b")
        axs[1, 0].plot(position[4], "b")
        axs[1, 1].plot(position[5], "b")
        axs[1, 2].plot(position[6], "b")
        if i % 10 == 0:
            print("Steps: ", i)
    axs[1, 3].plot(rewards)
    plt.show()


if __name__ == "__main__":
    try:
        # run the main execution
        main()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        # close sim app
        simulation_app.close()
