import gymnasium as gym
import torch
from fancy_gym.black_box.raw_interface_wrapper import RawInterfaceWrapper


class OrbitBlackBoxWrapper(RawInterfaceWrapper):

    def __init__(self, env):
        super().__init__(env)
        if self.env.spec.max_episode_steps is None:
            self.env.spec.max_episode_steps = self.env.unwrapped.max_episode_length

    mp_config = {
        "ProDMP": {
            "phase_generator_kwargs": {
                "phase_generator_type": "exp",
                "tau": 0.5,
            },
        }
    }

    @property
    def action_space(self):
        action_space = self.env.action_space
        return gym.spaces.Box(
            low=action_space.low[0],
            high=action_space.high[0],
        )

    @property
    def observation_space(self):
        if type(self.env.observation_space) == gym.spaces.Dict:
            key = list(self.env.observation_space.spaces.keys())[0]
            observation_space = self.env.observation_space[key]
        return gym.spaces.Box(
            low=observation_space.low[0],
            high=observation_space.high[0],
        )

    @property
    def dt(self):
        return self.env.unwrapped.step_dt

    @property
    def context_mask(self):
        # If the env already defines a context_mask, we will use that
        if hasattr(self.env, "context_mask"):
            return self.env.context_mask

        # Otherwise we will use the whole observation as the context. (Write a custom MPWrapper to change this behavior)
        return torch.full(self.observation_space.shape, True)

    @property
    def current_pos(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def current_vel(self) -> torch.Tensor:
        raise NotImplementedError
