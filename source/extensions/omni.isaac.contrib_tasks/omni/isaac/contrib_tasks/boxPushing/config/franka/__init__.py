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
for reward_type in ["Dense", "TemporalSparse"]:  # TODO add TemporalSpatialSparse
    for rsl_rl_cfg_name in ["Step_RL_Orbit_HP", "BBRL_Orbit_HP", "Step_RL_Fancy_Gym_HP", "BBRL_Fancy_Gym_HP"]:
        gym.register(
            id=f"Isaac-Box-Pushing-{reward_type}-{rsl_rl_cfg_name}-Franka-v0",
            entry_point="omni.isaac.contrib_tasks.boxPushing.box_pushing_env:BoxPushingEnv",
            kwargs={
                "env_cfg_entry_point": getattr(joint_pos_env_cfg, f"FrankaBoxPushingEnvCfg_{reward_type}"),
                "rsl_rl_cfg_entry_point": getattr(agents.rsl_rl_cfg, f"BoxPushingPPORunnerCfg_{rsl_rl_cfg_name}"),
            },
            disable_env_checker=True,
        )

    fancy_gym_registry.upgrade(
        id=f"Isaac-Box-Pushing-{reward_type}-{rsl_rl_cfg_name}-Franka-v0",
        mp_wrapper=joint_pos_env_cfg.FrankaBoxPushingMPWrapper,
    )
