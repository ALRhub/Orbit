# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.orbit.assets import Articulation, RigidObject
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.sensors import FrameTransformer
from omni.isaac.orbit.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import RLTaskEnv


def object_ee_distance(
    env: RLTaskEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return torch.clamp(object_ee_distance, min=0.05, max=100)


def object_goal_position_distance(
    env: RLTaskEnv,
    command_name: str,
    end_ep: bool,
    end_ep_weight: float = 0.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, t12=command[:, :3])
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)

    #  If there is a different weighting only to be computed at the end of an episode
    if end_ep:
        #  compute only for terminated envs
        terminated = env.termination_manager.dones
        distance += torch.where(terminated, distance, 0.0) * end_ep_weight
    return distance


def object_goal_orientation_distance(
    env: RLTaskEnv,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired orientation in the world frame
    _, des_or_w = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, q12=command[:, 3:])
    # distance of the end-effector to the object: (num_envs,)
    # rot_dist = rotation_distance(des_or_w, object.data.root_quat_w) / torch.pi
    rot_dist = yaw_rotation_distance(des_or_w, object.data.root_quat_w) / torch.pi
    return rot_dist


# TODO somehow asset_cfg.joint_ids is None so has to be replaced with :
def joint_pos_limits_bp(env: RLTaskEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions if they cross the soft limits.

    This is computed as a sum of the absolute value of the difference between the joint position and the soft limits.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    out_of_limits = -(asset.data.joint_pos[:, :] - asset.data.soft_joint_pos_limits[:, :, 0]).clip(max=0.0)
    out_of_limits += (asset.data.joint_pos[:, :] - asset.data.soft_joint_pos_limits[:, :, 1]).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


def end_ep_vel(
    env: RLTaskEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    #  retrieving velocity
    asset: Articulation = env.scene[asset_cfg.name]
    vel = torch.abs(asset.data.joint_vel[:, :7])

    reward = torch.norm(vel, dim=1)

    #  compute only for terminated envs
    terminated = env.termination_manager.dones
    reward = torch.where(terminated, reward, 0.0)

    return reward


def joint_vel_limits_bp(
    env: RLTaskEnv,
    soft_ratio: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint velocities if they cross the soft limits.

    This is computed as a sum of the absolute value of the difference between the joint velocity and the soft limits.

    Args:
        soft_ratio: The ratio of the soft limits to be used.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # max joint velocities
    arm_dof_vel_max = torch.tensor([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100], device=env.device)
    # compute out of limits constraints
    out_of_limits = torch.abs(asset.data.joint_vel[:, :7]) - arm_dof_vel_max * soft_ratio
    # clip to max error = 1 rad/s per joint to avoid huge penalties
    out_of_limits = out_of_limits.clip_(min=0.0, max=1.0)
    return torch.sum(out_of_limits, dim=1)


def rod_inclined_angle(
    env: RLTaskEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    desired_rod_quat = torch.tensor([0.0, 1.0, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1)
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_quat = ee_frame.data.target_quat_w[..., 0, :]
    rot_dist = rotation_distance(desired_rod_quat, ee_quat)
    return torch.where(rot_dist > torch.pi / 4.0, rot_dist / torch.pi, rot_dist * 0)


def rotation_distance(quat_a: torch.Tensor, quat_b: torch.Tensor) -> torch.Tensor:
    assert quat_a.shape == quat_b.shape
    theta = 2 * torch.acos(torch.abs(torch.einsum("ij,ij->i", quat_a, quat_b).unsqueeze(1)))
    return theta.squeeze()


def yaw_rotation_distance(quat_a: torch.Tensor, quat_b: torch.Tensor) -> torch.Tensor:
    assert quat_a.shape == quat_b.shape
    yaw_a = quaternion_to_axis_angle(quat_a)[:, 2]
    yaw_b = quaternion_to_axis_angle(quat_b)[:, 2]
    return torch.abs(yaw_a - yaw_b)


# From pytorch3d
def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = 0.5 - (angles[small_angles] * angles[small_angles]) / 48
    return quaternions[..., 1:] / sin_half_angles_over_angles
