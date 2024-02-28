# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for pose tracking."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from omni.isaac.orbit.assets import RigidObject
from omni.isaac.orbit.envs.mdp.commands.pose_command import UniformPoseCommand
from omni.isaac.orbit.markers import VisualizationMarkers
from omni.isaac.orbit.markers.config import FRAME_MARKER_CFG
from omni.isaac.orbit.utils.math import combine_frame_transforms, compute_pose_error, quat_from_euler_xyz

from omni.isaac.contrib_tasks.boxPushing.mdp.rewards import object_goal_distance

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import BaseEnv

    from .pose_command_min_dist_cfg import UniformPoseWithMinDistCommandCfg


class UniformPoseWithMinDistCommand(UniformPoseCommand):
    """Command generator for generating pose commands uniformly.

    The command generator generates poses by sampling positions uniformly within specified
    regions in cartesian space. For orientation, it samples uniformly the euler angles
    (roll-pitch-yaw) and converts them into quaternion representation (w, x, y, z).

    The position and orientation commands are generated in the base frame of the robot, and not the
    simulation world frame. This means that users need to handle the transformation from the
    base frame to the simulation world frame themselves.

    .. caution::

        Sampling orientations uniformly is not strictly the same as sampling euler angles uniformly.
        This is because rotations are defined by 3D non-Euclidean space, and the mapping
        from euler angles to rotations is not one-to-one.

    """

    cfg: UniformPoseWithMinDistCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: UniformPoseWithMinDistCommandCfg, env: BaseEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        self.box: RigidObject = env.scene[cfg.box_name]

        self.min_dist = cfg.min_dist

        self.max_iters = cfg.max_iters

    def __str__(self) -> str:
        msg = "UniformPoseCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired pose command. Shape is (num_envs, 7).

        The first three elements correspond to the position, followed by the quaternion orientation in (w, x, y, z).
        """
        return self.pose_command_b

    """
    Implementation specific functions.
    """

    def _resample_command(self, env_ids: Sequence[int]):

        box_position_b = self.box.data.root_pos_w[:, :3] - self.robot.data.root_state_w[:, :3]

        # sample new pose targets
        # -- position
        r = torch.empty(len(env_ids), device=self.device)
        self.pose_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.pos_x)
        self.pose_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.pos_y)
        self.pose_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.pos_z)
        
        for _ in range(self.max_iters):
            distances = torch.norm(self.pose_command_b[env_ids, :3] - box_position_b[env_ids, :], dim=1)
            mask = distances >= self.min_dist

            if mask.all():
                break

            resampled = self.resample_position(env_ids)
            self.pose_command_b[env_ids, :] = torch.where(mask.unsqueeze(1), self.pose_command_b[env_ids, :], resampled[env_ids, :])

        # -- orientation
        euler_angles = torch.zeros_like(self.pose_command_b[env_ids, :3])
        euler_angles[:, 0].uniform_(*self.cfg.ranges.roll)
        euler_angles[:, 1].uniform_(*self.cfg.ranges.pitch)
        euler_angles[:, 2].uniform_(*self.cfg.ranges.yaw)
        self.pose_command_b[env_ids, 3:] = quat_from_euler_xyz(
            euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2]
        )

    def resample_position(self, env_ids) -> torch.Tensor:
        r = torch.empty(len(env_ids), device=self.device)
        resampled = torch.zeros_like(self.pose_command_b)
        resampled[env_ids, 0] = r.uniform_(*self.cfg.ranges.pos_x)
        resampled[env_ids, 1] = r.uniform_(*self.cfg.ranges.pos_y)
        resampled[env_ids, 2] = r.uniform_(*self.cfg.ranges.pos_z)
        return resampled

    def _update_command(self):
        pass

    def _update_metrics(self):
        # transform command from base frame to simulation world frame
        self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = combine_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.pose_command_b[:, :3],
            self.pose_command_b[:, 3:],
        )
        # compute the error
        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            self.robot.data.body_state_w[:, self.body_idx, :3],
            self.robot.data.body_state_w[:, self.body_idx, 3:7],
        )
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                marker_cfg = FRAME_MARKER_CFG.copy()
                marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_pose"
                self.goal_pose_visualizer = VisualizationMarkers(marker_cfg)
                # -- current body pose
                marker_cfg.prim_path = "/Visuals/Command/body_pose"
                self.body_pose_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.goal_pose_visualizer.set_visibility(True)
            self.body_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.body_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        # -- goal pose
        self.goal_pose_visualizer.visualize(self.pose_command_w[:, :3], self.pose_command_w[:, 3:])
        # -- current body pose
        body_pose_w = self.robot.data.body_state_w[:, self.body_idx]
        self.body_pose_visualizer.visualize(body_pose_w[:, :3], body_pose_w[:, 3:7])
