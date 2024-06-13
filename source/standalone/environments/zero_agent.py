# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

from __future__ import annotations

from omni.isaac.orbit.controllers import DifferentialIKController, DifferentialIKControllerCfg

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.orbit.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Zero agent for Orbit environments.")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--motion_primitive",
    type=str,
    default=None,
    help=(
        "Whether to use a motion primitive for the training. The supported ones depend in the environment: ProDMP,"
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
import torch

import omni.isaac.contrib_tasks  # noqa: F401
import omni.isaac.orbit_tasks  # noqa: F401
from omni.isaac.orbit_tasks.utils import parse_env_cfg

from omni.isaac.orbit.markers.config import FRAME_MARKER_CFG  # isort: skip
from omni.isaac.orbit.markers import VisualizationMarkers  # isort: skip


def main():
    """Zero actions agent with Orbit environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
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

    target_pose_vis_frame = FRAME_MARKER_CFG.copy()
    target_pose_vis_frame.markers["frame"].scale = (0.1, 0.1, 0.1)
    target_pose_vis_frame.prim_path = "/Visuals/FrameTransformerEEFPose"
    marker = VisualizationMarkers(target_pose_vis_frame)

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # compute zero actions
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            if args_cli.motion_primitive is not None:
                actions = torch.zeros((args_cli.num_envs, env.action_space.shape[0]), device=env.unwrapped.device)
            # apply actions
            env.step(actions)
            marker.visualize(
                translations=torch.cat((env.unwrapped.EEF_target_poses[:, :3], env.unwrapped.EEF_poses[:, :3]), 0),
                orientations=torch.cat((env.unwrapped.EEF_target_poses[:, 3:7], env.unwrapped.EEF_poses[:, 3:7]), 0),
                # translations=env.unwrapped.EEF_poses[:, :3],
                # orientations=env.unwrapped.EEF_poses[:, 3:7],
            )

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
