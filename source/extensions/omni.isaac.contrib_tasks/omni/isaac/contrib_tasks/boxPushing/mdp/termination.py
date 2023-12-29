# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.orbit.assets import RigidObject
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.utils.math import combine_frame_transforms
from .rewards import object_goal_distance

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import RLTaskEnv


def is_success(
    env: RLTaskEnv,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    return object_goal_distance(env, command_name, robot_cfg, object_cfg) < 0.05