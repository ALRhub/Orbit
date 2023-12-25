# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import gymnasium as gym

from . import agents, ik_abs_env_cfg, ik_rel_env_cfg, joint_pos_env_cfg

##
# Register Gym environments.
##

##
# Joint Position Control
##

gym.register(
    id="Isaac-Push-Box-Franka-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    kwargs={
        "env_cfg_entry_point": joint_pos_env_cfg.FrankaBoxPushingEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.BoxPushingPPORunnerCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Push-Box-Franka-Play-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    kwargs={
        "env_cfg_entry_point": joint_pos_env_cfg.FrankaBoxPushingEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.BoxPushingPPORunnerCfg,
    },
    disable_env_checker=True,
)

##
# Inverse Kinematics - Absolute Pose Control
##

gym.register(
    id="Isaac-Push-Box-Franka-IK-Abs-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    kwargs={
        "env_cfg_entry_point": ik_abs_env_cfg.FrankaBoxPushingEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.BoxPushingPPORunnerCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Push-Box-Franka-IK-Abs-Play-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    kwargs={
        "env_cfg_entry_point": ik_abs_env_cfg.FrankaBoxPushingEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.BoxPushingPPORunnerCfg,
    },
    disable_env_checker=True,
)

##
# Inverse Kinematics - Relative Pose Control
##

gym.register(
    id="Isaac-Push-Box-Franka-IK-Rel-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    kwargs={
        "env_cfg_entry_point": ik_rel_env_cfg.FrankaBoxPushingEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.BoxPushingPPORunnerCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Push-Box-Franka-IK-Rel-Play-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    kwargs={
        "env_cfg_entry_point": ik_rel_env_cfg.FrankaBoxPushingEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.BoxPushingPPORunnerCfg,
    },
    disable_env_checker=True,
)