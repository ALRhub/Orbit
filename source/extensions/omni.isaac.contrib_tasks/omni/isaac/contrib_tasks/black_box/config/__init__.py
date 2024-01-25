# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import gymnasium as gym

from . import agents, box_pushing_bb_cfg

##
# Register Gym environments.
##

##
# Joint Position Control
##

gym.register(
    id="Isaac-BB-Push-Box-Franka-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    kwargs={
        "env_cfg_entry_point": box_pushing_bb_cfg.BoxPushingBB,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.BoxPushingPPORunnerCfg,
    },
    disable_env_checker=True,
)

# gym.register(
#     id="Isaac-Push-Box-Franka-Play-v0",
#     entry_point="omni.isaac.orbit.envs:RLTaskEnv",
#     kwargs={
#         "env_cfg_entry_point": box_pushing_bb_cfg.FrankaBoxPushingEnvCfg_PLAY,
#         "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.BoxPushingPPORunnerCfg,
#     },
#     disable_env_checker=True,
# )
