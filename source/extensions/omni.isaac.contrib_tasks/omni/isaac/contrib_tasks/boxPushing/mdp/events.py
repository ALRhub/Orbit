# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.orbit.assets import Articulation, RigidObject
from omni.isaac.orbit.controllers import DifferentialIKController, DifferentialIKControllerCfg
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.utils.math import quat_from_euler_xyz, sample_uniform, subtract_frame_transforms

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import BaseEnv


def reset_box_root_state_uniform_with_robot_IK(
    env: BaseEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root state to a random position and velocity uniformly within the given ranges.

    This function randomizes the root position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default root position, before setting
      them into the physics simulation.
    * It samples the root orientation from the given ranges and sets them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of position and velocity ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or velocity is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()

    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientations = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])

    # set into the physics simulation
    box_poses = torch.cat([positions, orientations], dim=-1)
    asset.write_root_pose_to_sim(box_poses, env_ids=env_ids)

    robot: Articulation = env.scene["robot"]

    # processing target box poses
    target_poses = box_poses + torch.tensor([0.0, 0.0, 0.27, 0.0, 0.0, 0.0, 0.0], device=env.device)
    target_poses[:, :3] -= robot.data.root_pos_w
    target_poses[:, 3:7] = torch.tensor([0.0, 1.0, 0.0, 0.0], device=env.device)

    # TODO test IK computing
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=env.scene.num_envs, device=env.device)
    # reset controller
    diff_ik_controller.reset()
    diff_ik_controller.set_command(target_poses, ee_quat=target_poses[:, 3:7])
    # Specify robot-specific parameters
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    robot_entity_cfg.resolve(env.scene)
    # Obtain the frame index of the end-effector
    # For a fixed base robot, the frame index is one less than the body index. This is because
    # the root body is not included in the returned Jacobians.

    if robot.is_fixed_base:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
    else:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0]

    # obtain quantities from simulation
    jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
    ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
    root_pose_w = robot.data.root_state_w[:, :7]
    joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
    # compute frame in root frame
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        root_pose_w[:, :3], root_pose_w[:, 3:7], ee_pose_w[:, :3], ee_pose_w[:, 3:7]
    )

    # TODO remove
    target_poses = torch.cat((ee_pos_b, ee_quat_b), 1)
    diff_ik_controller.reset()
    diff_ik_controller.set_command(target_poses, ee_quat=target_poses[:, 3:7])

    # For broadcasting reasons
    env_ids = env_ids[:, None]

    # compute the joint commands
    joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)[
        env_ids, robot_entity_cfg.joint_ids
    ]

    # robot.data.default_joint_pos[:, :7] = joint_pos_des

    robot.write_joint_state_to_sim(
        joint_pos_des,
        torch.zeros(joint_pos_des.shape, device=env.device),
        joint_ids=robot_entity_cfg.joint_ids,
        env_ids=env_ids,
    )

    ee_pose_w_cpy = ee_pose_w.clone()

    # TODO remove
    # for i in range(5):
    #     env.step(torch.zeros(env.action_space.shape, device=env.device))
    ee_pose_w_act = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], :7]
    env.EEF_poses = ee_pose_w_act.clone()
    env.EEF_target_poses = target_poses
    env.EEF_target_poses[:, :3] += robot.data.root_pos_w
