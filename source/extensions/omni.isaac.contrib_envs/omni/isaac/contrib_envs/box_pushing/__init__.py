# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment for lifting objects with fixed-arm robots."""

from .box_pushing_cfg import BoxPushingEnvCfg
from .box_pushing_env import BoxPushingEnv

__all__ = ["BoxPushingEnv", "BoxPushingEnvCfg"]
