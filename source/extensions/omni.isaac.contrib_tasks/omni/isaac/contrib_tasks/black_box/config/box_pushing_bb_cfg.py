# File containing the properties of the mps for each step based env

import numpy as np

from typing import Union, Tuple

import gym
from omni.isaac.orbit.utils import configclass

# TODO just pasted the code from fancy_gym but already changed the class
# ned to adapt the class to finalize the initialization


@configclass
class MPWrapper(gym.Wrapper):

    #TODO out init of the super obj and of the configs
    mp_config = {
        'ProMP': {
            'controller_kwargs': {
                'p_gains': 0.01 * np.array([120., 120., 120., 120., 50., 30., 10.]),
                'd_gains': 0.01 * np.array([10., 10., 10., 10., 6., 5., 3.]),
            },
            'basis_generator_kwargs': {
                'basis_bandwidth_factor': 2  # 3.5, 4 to try
            }
        },
        'DMP': {},
        'ProDMP': {
            'controller_kwargs': {
                'p_gains': 0.01 * np.array([120., 120., 120., 120., 50., 30., 10.]),
                'd_gains': 0.01 * np.array([10., 10., 10., 10., 6., 5., 3.]),
            },
            'basis_generator_kwargs': {
                'basis_bandwidth_factor': 2  # 3.5, 4 to try
            }
        },
    }

    # Random x goal + random init pos
    @property
    def context_mask(self):
        if self.random_init:
            return np.hstack([
                [True] * 7,  # joints position
                [False] * 7,  # joints velocity
                [True] * 3,  # position of box
                [True] * 4,  # orientation of box
                [True] * 3,  # position of target
                [True] * 4,  # orientation of target
                # [True] * 1,  # time
            ])

        return np.hstack([
            [False] * 7,  # joints position
            [False] * 7,  # joints velocity
            [False] * 3,  # position of box
            [False] * 4,  # orientation of box
            [True] * 3,  # position of target
            [True] * 4,  # orientation of target
            # [True] * 1,  # time
        ])

    @property
    def current_pos(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.data.qpos[:7].copy()

    @property
    def current_vel(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.data.qvel[:7].copy()