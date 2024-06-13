# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from dataclasses import MISSING

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from omni.isaac.orbit.envs import RLTaskEnvCfg
from omni.isaac.orbit.managers import EventTermCfg as EventTerm
from omni.isaac.orbit.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.orbit.managers import ObservationTermCfg as ObsTerm
from omni.isaac.orbit.managers import RewardTermCfg as RewTerm
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.managers import TerminationTermCfg as DoneTerm
from omni.isaac.orbit.scene import InteractiveSceneCfg
from omni.isaac.orbit.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from omni.isaac.orbit.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR

from omni.isaac.contrib_tasks.boxPushing.mdp.commands.pose_command_min_dist_cfg import UniformPoseWithMinDistCommandCfg
from omni.isaac.contrib_tasks.boxPushing.mdp.events import reset_box_root_state_uniform_with_robot_IK

from . import mdp

##
# Scene definition
##


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING
    # target object: will be populated by agent env cfg
    object: RigidObjectCfg = MISSING

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
            scale=(1.5, 1.0, 1.0),
        ),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose = UniformPoseWithMinDistCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # will be set by agent env cfg
        box_name="object",
        min_dist=0.15,
        resampling_time_range=(10.0, 10.0),
        debug_vis=True,
        ranges=UniformPoseWithMinDistCommandCfg.Ranges(
            pos_x=(0.3, 0.6),
            pos_y=(-0.45, 0.45),
            pos_z=(0.007, 0.007),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0, 2 * torch.pi),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    body_joint_pos: mdp.JointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # joint_pos_abs = ObsTerm(func=mdp.joint_pos_abs)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_pose = ObsTerm(func=mdp.object_pose_in_robot_root_frame)
        target_object_pose = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for randomization."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=reset_box_root_state_uniform_with_robot_IK,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )


@configclass
class DenseRewardCfg:
    """Reward terms for the MDP."""

    object_ee_distance = RewTerm(func=mdp.object_ee_distance, weight=-1.0)

    object_goal_position_distance = RewTerm(
        func=mdp.object_goal_position_distance,
        params={"end_ep": False, "end_ep_weight": 100.0, "command_name": "object_pose"},
        weight=-3.5,
    )

    object_goal_orientation_distance = RewTerm(
        func=mdp.object_goal_orientation_distance,
        params={"command_name": "object_pose"},
        weight=-2.0,
    )

    energy_cost = RewTerm(func=mdp.action_l2, weight=-5e-4)

    joint_position_limit = RewTerm(func=mdp.joint_pos_limits_bp, weight=-1.0)

    joint_velocity_limit = RewTerm(func=mdp.joint_vel_limits_bp, params={"soft_ratio": 1.0}, weight=-1.0)

    rod_inclined_angle = RewTerm(func=mdp.rod_inclined_angle, weight=-1.0)


@configclass
class TemporalSparseRewardCfg:  # TODO set weights
    """Reward terms for the MDP."""

    object_ee_distance = RewTerm(func=mdp.object_ee_distance, weight=-1.0)

    object_goal_position_distance = RewTerm(
        func=mdp.object_goal_position_distance,
        params={"end_ep": True, "end_ep_weight": 100.0, "command_name": "object_pose"},
        weight=-3.5,
    )

    object_goal_orientation_distance = RewTerm(
        func=mdp.object_goal_orientation_distance,
        params={"command_name": "object_pose"},
        weight=-2.0,
    )

    energy_cost = RewTerm(func=mdp.action_l2, weight=-5e-4)

    joint_position_limit = RewTerm(func=mdp.joint_pos_limits_bp, weight=-1.0)

    joint_velocity_limit = RewTerm(func=mdp.joint_vel_limits_bp, params={"soft_ratio": 1.0}, weight=-1.0)

    rod_inclined_angle = RewTerm(func=mdp.rod_inclined_angle, weight=-1.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    success = DoneTerm(
        func=mdp.is_success, params={"command_name": "object_pose", "limit_pose_dist": 0.05, "limit_or_dist": 0.5}
    )


##
# Environment configuration
##


@configclass
class BoxPushingEnvCfg(RLTaskEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    # rewards: will be populated by agent env cfg
    rewards = MISSING
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""

        # simulation settings
        self.sim.dt = 0.01  # 100Hz

        # general settings
        max_steps = 200
        self.decimation = 2
        self.episode_length_s = max_steps * self.sim.dt

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 50 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
