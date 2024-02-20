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
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
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
import numpy as np
import torch
import traceback

import carb

import omni.isaac.contrib_tasks  # noqa: F401
import omni.isaac.orbit_tasks  # noqa: F401
from omni.isaac.orbit_tasks.utils import parse_env_cfg


def main():
    """Random actions agent with Orbit environment."""
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs)
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
            actions = torch.zeros(
                (args_cli.num_envs, env.action_space.shape[0]),
                device=env.unwrapped.device,
            )
            action = np.around(np.load("/home/johann/random_sample_numpy.npy"), 4)
            for i in range(args_cli.num_envs):
                actions[i] = torch.from_numpy(action)
                # actions[i] = torch.load("/home/johann/random_sample_torch.pt")
            # apply actions
            obs, reward, _, _, info = env.step(actions)
            # torch.save(info["positions"][0], "/home/johann/traj_gen_orbit.pt")
            rewards.append(reward[0].cpu().numpy())
            infos.append(info)
    # close the simulator
    plot_trajectories(env, infos)
    env.close()


def plot_trajectories(env, infos, rewards=None):
    import pickle

    file = open("/home/johann/positions.pkl", "rb")
    data = pickle.load(file)

    from omni.isaac.orbit.assets import Articulation

    asset: Articulation = env.unwrapped.scene["robot"]
    orbit_joint_pos_limit = asset.data.soft_joint_pos_limits[0].cpu().tolist()

    xdim = 2
    ydim = 4
    _, axs = plt.subplots(xdim, ydim)
    mse = []
    for i in range(len(data)):
        ref_traj = infos[i]["positions"][0].transpose(0, 1).cpu().tolist()
        orbit_joint_obs = infos[i]["step_observations"][0, :, :7].transpose(0, 1).cpu().tolist()
        fancy_gym_joint_obs = data[i]["step_observations"][:, :7].transpose().tolist()
        for i in range(xdim):
            for j in range(ydim):
                if i != 1 or j != 3:
                    index = i * 4 + j
                    axs[i, j].plot(ref_traj[index], "b")
                    axs[i, j].plot(orbit_joint_obs[index], "r")
                    axs[i, j].plot(fancy_gym_joint_obs[index], "g")
                    # axs[i, j].hlines(y=orbit_joint_pos_limit[index], xmin=0, xmax=101, color='y')
        mse.append(np.square(np.array(ref_traj) - np.array(orbit_joint_obs)).mean(axis=1))
    print("MSE orbit: ", np.array(mse).mean(axis=0))
    if rewards:
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
