# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import gymnasium as gym

import fancy_gym.envs.registry as fancy_gym_registry

from . import agents, joint_pos_env_cfg

##
# Register Gym environments.
##

##
# Joint Position Control
##

# Dense reward
for reward_type in ["Dense", "TemporalSparse"]: #TODO add TemporalSpatialSparse
    gym.register(
        id="Isaac-Box-Pushing-{}-Franka-v0".format(reward_type),
        entry_point="omni.isaac.orbit.envs:RLTaskEnv",
        kwargs={
            "env_cfg_entry_point": getattr(joint_pos_env_cfg, "FrankaBoxPushingEnvCfg_{}".format(reward_type)),
            "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.BoxPushingPPORunnerCfg,
        },
        disable_env_checker=True,
    )

fancy_gym_registry.upgrade(
    id="Isaac-Box-Pushing-Franka-v0",
    mp_wrapper=joint_pos_env_cfg.FrankaBoxPushingMPWrapper,
)
