# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.orbit.controllers.differential_inverse_kinematics import DifferentialInverseKinematicsCfg
from omni.isaac.orbit.objects import RigidObjectCfg
from omni.isaac.orbit.robots.config.franka import FRANKA_PANDA_ARM_WITH_PANDA_HAND_CFG
from omni.isaac.orbit.robots.single_arm import SingleArmManipulatorCfg
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR

from omni.isaac.orbit_envs.isaac_env_cfg import EnvCfg, IsaacEnvCfg, PhysxCfg, SimCfg, ViewerCfg

import os

##
# Scene settings
##

#TODO Change scene to resemble mujoco scene
@configclass
class TableCfg:
    """Properties for the table."""

    # note: we use instanceable asset since it consumes less memory
    usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"

#TODO Change scene to resemble mujoco scene
@configclass
class ManipulationObjectCfg(RigidObjectCfg):
    """Properties for the object to manipulate in the scene."""

    meta_info = RigidObjectCfg.MetaInfoCfg(
        usd_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets/box_cad.usd"),
        scale=(0.01, 0.01, 0.01),
    )
    init_state = RigidObjectCfg.InitialStateCfg(
        pos=(0.4, 0.0, 0.075), rot=(1.0, 0.0, 0.0, 0.0), lin_vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0)
    )
    rigid_props = RigidObjectCfg.RigidBodyPropertiesCfg(
        solver_position_iteration_count=16,
        solver_velocity_iteration_count=1,
        max_angular_velocity=1000.0,
        max_linear_velocity=1000.0,
        max_depenetration_velocity=5.0,
        disable_gravity=False,
    )
    physics_material = RigidObjectCfg.PhysicsMaterialCfg(
        static_friction=0.5, dynamic_friction=0.5, restitution=0.0, prim_path="/World/Materials/cubeMaterial"
    )

@configclass
class GoalMarkerCfg:
    """Properties for visualization marker."""

    # usd file to import
    usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd"
    # scale of the asset at import
    scale = [0.05, 0.05, 0.05]  # x,y,z

@configclass
class FrameMarkerCfg:
    """Properties for visualization marker."""

    # usd file to import
    usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd"
    # scale of the asset at import
    scale = [0.1, 0.1, 0.1]  # x,y,z


##
# MDP settings
##

#TODO visualise limits and adapt them (after recreating the mujoco scene)
@configclass
class RandomizationCfg:
    """Randomization of scene at reset."""

    @configclass
    class ObjectInitialPoseCfg:
        """Randomization of object initial pose."""

        # category
        position_cat: str = "default"  # randomize position: "default", "uniform"
        orientation_cat: str = "default"  # randomize position: "default", "uniform"
        # randomize position
        position_uniform_min = [0.4, -0.25, 0.075]  # position (x,y,z)
        position_uniform_max = [0.6, 0.25, 0.075]  # position (x,y,z)

    @configclass
    class ObjectDesiredPoseCfg:
        """Randomization of object desired pose."""

        # category
        position_cat: str = "uniform"  # randomize position: "default", "uniform"
        orientation_cat: str = "uniform"  # randomize position: "default", "uniform"
        # randomize position
        position_default = [0.5, 0.0, 0.0]  # position default (x,y,z)
        position_uniform_min = [0.4, -0.4, 0.0]  # position (x,y,z)
        position_uniform_max = [0.6, 0.4, 0.0]  # position (x,y,z)
        # randomize orientation
        orientation_default = [1.0, 0.0, 0.0, 0.0]  # orientation default

    # initialize
    object_initial_pose: ObjectInitialPoseCfg = ObjectInitialPoseCfg()
    object_desired_pose: ObjectDesiredPoseCfg = ObjectDesiredPoseCfg()

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg:
        """Observations for policy group."""

        # global group settings
        enable_corruption: bool = True
        # observation terms
        # -- joint state
        arm_dof_pos = {"scale": 1.0}
        # arm_dof_pos_scaled = {"scale": 1.0}
        arm_dof_vel = {"scale": 1.0, "noise": {"name": "uniform", "min": -0.01, "max": 0.01}}
        # tool_dof_pos_scaled = {"scale": 1.0}
        # -- end effector state
        tool_positions = {"scale": 1.0}
        tool_orientations = {"scale": 1.0}
        # -- object state
        object_positions = {"scale": 1.0}
        object_orientations = {"scale": 1.0}
        # object_relative_tool_positions = {"scale": 1.0}
        # object_relative_tool_orientations = {"scale": 1.0}
        # -- object desired state
        object_desired_positions = {"scale": 1.0}
        object_desired_orientations = {"scale": 1.0}
        # -- previous action
        arm_actions = {"scale": 1.0}
        # tool_actions = {"scale": 1.0}

    # global observation settings
    return_dict_obs_in_group = False
    """Whether to return observations as dictionary or flattened vector within groups."""
    # observation groups
    policy: PolicyCfg = PolicyCfg()

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    box_pushing_dense = {"weight": 1.0}
    
    # Deactivated (weight = 0)
    box_pushing_temporal_sparse = {"weight": 0.0}
    box_pushing_temporal_spatial_sparse = {"weight": 0.0}

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    episode_timeout = True  # reset when episode length ended
    object_falling = True  # reset when object falls off the table
    is_success = False  # reset when object is lifted

@configclass
class ControlCfg:
    """Processing of MDP actions."""

    # action space
    control_type = "inverse_kinematics"  # "default", "inverse_kinematics"
    # decimation: Number of control action updates @ sim dt per policy dt
    decimation = 2

    # configuration loaded when control_type == "inverse_kinematics"
    inverse_kinematics: DifferentialInverseKinematicsCfg = DifferentialInverseKinematicsCfg(
        command_type="pose_rel",
        ik_method="dls",
        position_command_scale=(0.1, 0.1, 0.1),
        rotation_command_scale=(0.1, 0.1, 0.1),
    )


##
# Environment configuration
##

#TODO change parameters like episode length
@configclass
class BoxPushingEnvCfg(IsaacEnvCfg):
    """Configuration for the Lift environment."""

    # General Settings
    sim_step_size = 0.01
    max_steps = 100
    episode_length_s = max_steps * sim_step_size
    env: EnvCfg = EnvCfg(num_envs=4096, env_spacing=2.5, episode_length_s=episode_length_s)
    viewer: ViewerCfg = ViewerCfg(debug_vis=True, eye=(7.5, 7.5, 7.5), lookat=(0.0, 0.0, 0.0))
    # Physics settings
    sim: SimCfg = SimCfg(
        dt=sim_step_size,
        substeps=1,
        physx=PhysxCfg(
            gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
            gpu_total_aggregate_pairs_capacity=16 * 1024,
            friction_correlation_distance=0.00625,
            friction_offset_threshold=0.01,
            bounce_threshold_velocity=0.2,
        ),
    )

    # Scene Settings
    # -- robot
    robot: SingleArmManipulatorCfg = FRANKA_PANDA_ARM_WITH_PANDA_HAND_CFG
    # -- object
    object: ManipulationObjectCfg = ManipulationObjectCfg()
    # -- table
    table: TableCfg = TableCfg()
    # -- visualization marker
    goal_marker: GoalMarkerCfg = GoalMarkerCfg()
    frame_marker: FrameMarkerCfg = FrameMarkerCfg()

    # MDP settings
    randomization: RandomizationCfg = RandomizationCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Controller settings
    control: ControlCfg = ControlCfg()
