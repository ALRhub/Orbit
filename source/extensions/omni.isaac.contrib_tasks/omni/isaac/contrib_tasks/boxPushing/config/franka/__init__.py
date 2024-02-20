# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import gymnasium as gym

import fancy_gym.envs.registry as fancy_gym_registry

from . import agents, ik_abs_env_cfg, ik_rel_env_cfg, joint_pos_env_cfg

##
# Register Gym environments.
##

##
# Joint Position Control
##

gym.register(
    id="Isaac-Box-Pushing-Franka-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    kwargs={
        "env_cfg_entry_point": joint_pos_env_cfg.FrankaBoxPushingEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.BoxPushingPPORunnerCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Box-Pushing-Franka-Play-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    kwargs={
        "env_cfg_entry_point": joint_pos_env_cfg.FrankaBoxPushingEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.BoxPushingPPORunnerCfg,
    },
    disable_env_checker=True,
)

fancy_gym_registry.upgrade(
    id="Isaac-Box-Pushing-Franka-v0",
    mp_wrapper=joint_pos_env_cfg.FrankaBoxPushingMPWrapper,
)

##
# Inverse Kinematics - Absolute Pose Control
##

gym.register(
    id="Isaac-Box-Pushing-Franka-IK-Abs-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    kwargs={
        "env_cfg_entry_point": ik_abs_env_cfg.FrankaBoxPushingEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.BoxPushingPPORunnerCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Box-Pushing-Franka-IK-Abs-Play-v0",
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
    id="Isaac-Box-Pushing-Franka-IK-Rel-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    kwargs={
        "env_cfg_entry_point": ik_rel_env_cfg.FrankaBoxPushingEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.BoxPushingPPORunnerCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Box-Pushing-Franka-IK-Rel-Play-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    kwargs={
        "env_cfg_entry_point": ik_rel_env_cfg.FrankaBoxPushingEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.BoxPushingPPORunnerCfg,
    },
    disable_env_checker=True,
)
