# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import torch
from collections.abc import Sequence

from omni.isaac.orbit.assets import Articulation
from omni.isaac.orbit.assets.rigid_object.rigid_object import RigidObject
from omni.isaac.orbit.controllers import DifferentialIKController, DifferentialIKControllerCfg
from omni.isaac.orbit.envs.rl_task_env import RLTaskEnv
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.utils.math import subtract_frame_transforms


class BoxPushingEnv(RLTaskEnv):

    EEF_target_poses = None
    EEF_poses = None
    diff = torch.zeros((4, 7), device=torch.device("cuda:0"))

    def compute_robot_init_config(self, env_ids, target_poses):
        robot: Articulation = self.scene["robot"]

        # TODO test IK computing
        diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
        diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=self.scene.num_envs, device=self.device)
        # reset controller
        diff_ik_controller.reset()
        diff_ik_controller.set_command(target_poses, ee_quat=target_poses[:, 3:7])
        # Specify robot-specific parameters
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
        robot_entity_cfg.resolve(self.scene)
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
        # target_poses = torch.cat((ee_pos_b, ee_quat_b), 1)
        # diff_ik_controller.reset()
        # diff_ik_controller.set_command(target_poses, ee_quat=target_poses[:, 3:7])

        # FOr broadcasting reasons
        env_ids = env_ids[:, None]

        # compute the joint commands
        joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)[
            env_ids, robot_entity_cfg.joint_ids
        ]

        robot.write_joint_state_to_sim(
            joint_pos_des,
            torch.zeros(joint_pos_des.shape, device=self.device),
            joint_ids=robot_entity_cfg.joint_ids,
            env_ids=env_ids,
        )

        ee_pose_w_cpy = ee_pose_w.clone()

        # TODO remove
        for i in range(5):
            self.step(torch.zeros(self.action_space.shape, device=self.device))
        joint_pos_act = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
        ee_pose_w_act = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], :7]
        joint_pos_diff_act = joint_pos - joint_pos_act
        joint_pos_diff_des = joint_pos_des - joint_pos
        eef_pose_diff = ee_pose_w_act - ee_pose_w_cpy
        self.diff += joint_pos_diff_act
        self.EEF_poses = ee_pose_w_act.clone()
        self.EEF_target_poses = target_poses
        self.EEF_target_poses[:, :3] += robot.data.root_pos_w

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)

        # box: RigidObject = self.scene["object"]
        # robot: Articulation = self.scene["robot"]
        # # frame = self.scene["ee_frame"]
        # # ee_poses = frame.data.target_pos_w
        # box_poses = box.data.root_pos_w - robot.data.root_pos_w
        # target_poses = box_poses + torch.tensor([0.0, 0.0, 0.27], device=self.device)
        # target_poses = torch.nn.functional.pad(target_poses, (0, 4), "constant", 0)
        # target_poses[:, 3:7] = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device)
        #
        # self.compute_robot_init_config(env_ids, target_poses)
