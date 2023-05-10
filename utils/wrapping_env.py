from typing import Tuple

import gym
from gym.core import ObsType, ActType
from gym.wrappers.time_limit import TimeLimit
from typing import Tuple


def all_none(a, b):
    if (a is None) and (b is None):
        return True
    else:
        return False


class ResetInfoData(gym.Wrapper):
    """
    This wrapper ensures that the 'reset' method returns a tuple (obs, info).
    """

    def reset(self, **kwargs) -> Tuple[ObsType, dict]:
        ret = self.env.reset(**kwargs)
        if isinstance(ret, tuple):
            return ret
        else:
            return ret, {}


class ShapingRewardData(gym.Wrapper):
    """
        r_rescaled = (r + reward_shift) * reward_scale
        info["raw_reward"] = r
        example: add following to example script
            parser.add_argument("--reward_scale", default=0.5)
            parser.add_argument("--reward_shift", default=0)
    """

    def __init__(self, env, reward_shift: float = 0.0, reward_scale: float = 1.0):
        super(ShapingRewardData, self).__init__(env)
        self.reward_shift = reward_shift
        self.reward_scale = reward_scale

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        obs, r, d, info = self.env.step(action)
        r_scaled = (r + self.reward_shift) * self.reward_scale
        info["raw_reward"] = r
        return obs, r_scaled, d, info


def wrapping_env(env,
                 max_episode_steps=None,
                 reward_shift=None,
                 reward_scale=None,
                 ):
    env = ResetInfoData(env)
    if max_episode_steps is None and hasattr(env, "max_episode_steps"):
        max_episode_steps = getattr(env, "max_episode_steps")
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps)

    if not all_none(reward_scale, reward_shift):
        reward_scale = 1.0 if reward_scale is None else reward_scale
        reward_shift = 0.0 if reward_shift is None else reward_shift
        env = ShapingRewardData(env, reward_shift, reward_scale)

    return env
