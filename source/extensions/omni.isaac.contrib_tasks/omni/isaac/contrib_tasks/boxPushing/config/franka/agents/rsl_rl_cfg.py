# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.orbit.utils import configclass

from omni.isaac.orbit_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

@configclass
class BoxPushingPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = -1
    num_steps_per_env = 24
    max_iterations = 800
    save_interval = 50
    experiment_name = "step_rl_Orbit_HP"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=10,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="fixed",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


# @configclass
# class BoxPushingPPORunnerCfg(RslRlOnPolicyRunnerCfg):  # TODO ProMP only
#     seed = -1
#     num_steps_per_env = 1
#     max_iterations = 3000
#     save_interval = 100
#     experiment_name = "bbrl_Orbit_HP"
#     empirical_normalization = False
#     policy = RslRlPpoActorCriticCfg(
#         init_noise_std=1.0,
#         actor_hidden_dims=[256, 128, 64],
#         critic_hidden_dims=[256, 128, 64],
#         activation="elu",
#     )
#     algorithm = RslRlPpoAlgorithmCfg(
#         value_loss_coef=1.0,
#         use_clipped_value_loss=True,
#         clip_param=0.2,
#         entropy_coef=0.006,
#         num_learning_epochs=100,
#         num_mini_batches=1,
#         learning_rate=1.0e-4,
#         schedule="fixed",
#         gamma=0.98,
#         lam=0.95,
#         desired_kl=0.01,
#         max_grad_norm=1.0,
#     )


# @configclass
# class BoxPushingPPORunnerCfg(RslRlOnPolicyRunnerCfg):
#     seed = -1
#     num_steps_per_env = 100
#     max_iterations = 2500
#     save_interval = 250
#     experiment_name = "step_rl_Fancy_Gym_HP"
#     empirical_normalization = True
#     policy = RslRlPpoActorCriticCfg(
#         init_noise_std=1.0,
#         actor_hidden_dims=[256, 256],
#         critic_hidden_dims=[256, 256],
#         activation="tanh",
#     )
#     algorithm = RslRlPpoAlgorithmCfg(
#         value_loss_coef=1.0,
#         use_clipped_value_loss=True,
#         clip_param=0.2,
#         entropy_coef=0.006,
#         num_learning_epochs=10,
#         num_mini_batches=40,
#         learning_rate=1.0e-4,
#         schedule="fixed",
#         gamma=0.99,
#         lam=0.95,
#         desired_kl=0.01,
#         max_grad_norm=1.0,
#     )


# @configclass
# class BoxPushingPPORunnerCfg(RslRlOnPolicyRunnerCfg):     # TODO ProMP only
#     seed = -1
#     num_steps_per_env = 1
#     max_iterations = 2500
#     save_interval = 250
#     experiment_name = "bbrl_Fancy_Gym_HP"
#     empirical_normalization = False
#     policy = RslRlPpoActorCriticCfg(
#         init_noise_std=1.0,
#         actor_hidden_dims=[128, 128],
#         critic_hidden_dims=[32, 32],
#         activation="tanh",
#     )
#     algorithm = RslRlPpoAlgorithmCfg(
#         value_loss_coef=1.0,
#         use_clipped_value_loss=True,
#         clip_param=0.2,
#         entropy_coef=0.006,
#         num_learning_epochs=100,
#         num_mini_batches=1,
#         learning_rate=1.0e-4,
#         schedule="fixed",
#         gamma=0.99,
#         lam=0.95,
#         desired_kl=0.01,
#         max_grad_norm=1.0,
#     )
