# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.orbit.managers import SceneEntityCfg

from .rewards import object_goal_orientation_distance, object_goal_position_distance

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import RLTaskEnv


def is_success(
    env: RLTaskEnv,
    command_name: str,
    limit_pose_dist: float = 0.05,
    limit_or_dist: float = 0.5,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    terminated = env.termination_manager.dones
    return torch.where(
        terminated,
        torch.logical_and(
            object_goal_position_distance(env, command_name, False, 0.0, robot_cfg, object_cfg) < limit_pose_dist,
            object_goal_orientation_distance(env, command_name, robot_cfg, object_cfg) < limit_or_dist,
        ),
        False,
    )
