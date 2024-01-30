# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

from omni.isaac.contrib_tasks.boxPushing.config.franka import orbit_black_box_wrapper

import numpy as np

import torch

from omni.isaac.orbit.utils import configclass

from omni.isaac.orbit.assets import RigidObjectCfg
from omni.isaac.orbit.sensors import FrameTransformerCfg
from omni.isaac.orbit.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from omni.isaac.orbit.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from omni.isaac.orbit.sim.spawners.from_files.from_files_cfg import UsdFileCfg

from omni.isaac.contrib_tasks.boxPushing.box_pushing_env_cfg import BoxPushingEnvCfg
from omni.isaac.orbit_tasks.manipulation.lift import mdp

from omni.isaac.orbit.assets import Articulation

##
# Pre-defined configs
##
from omni.isaac.orbit.markers.config import FRAME_MARKER_CFG  # isort: skip
from omni.isaac.contrib_tasks.boxPushing.assets.franka import FRANKA_PANDA_CFG  # isort: skip


@configclass
class FrankaBoxPushingEnvCfg(BoxPushingEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (franka)
        self.actions.body_joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )
        self.commands.object_pose.body_name = "panda_hand"

        # Set Cube as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../assets/box.usda"),
                scale=(0.001, 0.001, 0.001),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.27],
                    ),
                ),
            ],
        )


@configclass
class FrankaBoxPushingEnvCfg_PLAY(FrankaBoxPushingEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False


class FrankaBoxPushingMPWrapper(orbit_black_box_wrapper.OrbitBlackBoxWrapper):
    mp_config = {
        'ProMP': {
            'controller_kwargs': {
                'p_gains': 0.01 * torch.tensor([120., 120., 120., 120., 50., 30., 10.], device='cuda:0'),
                'd_gains': 0.01 * torch.tensor([10., 10., 10., 10., 6., 5., 3.], device='cuda:0'),
            },
            'basis_generator_kwargs': {
                'basis_bandwidth_factor': 2  # 3.5, 4 to try
            }
        },
        'DMP': {},
        'ProDMP': {
            'controller_kwargs': {
                'p_gains': 0.01 * torch.tensor([120., 120., 120., 120., 50., 30., 10.], device='cuda:0'),
                'd_gains': 0.01 * torch.tensor([10., 10., 10., 10., 6., 5., 3.], device='cuda:0'),
            },
            'basis_generator_kwargs': {
                'basis_bandwidth_factor': 2  # 3.5, 4 to try
            }
        },
    }

    # Random x goal + random init pos
    @property
    def context_mask(self):
        return np.hstack([
            [True] * 7,  # joints position
            [False] * 7,  # joints velocity
            [True] * 3,  # position of box
            [True] * 4,  # orientation of box
            [True] * 3,  # position of target
            [True] * 4,  # orientation of target
            # [True] * 1,  # time
            [True] * 7   # TODO wtf?
        ])

    @property
    def current_pos(self) -> torch.Tensor:
        scene = self.env.unwrapped.scene
        # return scene.rigid_objects["object"]._data.root_pos_w - scene.env_origins
        asset: Articulation = scene["robot"]
        return asset.data.joint_pos[:, :7]

    @property
    def current_vel(self) -> torch.Tensor:
        scene = self.env.unwrapped.scene
        asset: Articulation = scene["robot"]
        return asset.data.joint_vel[:, :7]
