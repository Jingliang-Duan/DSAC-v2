from typing import Optional,Union
import numpy as np
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

class StateData(gym.Wrapper):
    """
    Wrapper ensures that environment has "state" property.
    If original environment does not have one, current observation is returned when calling state.
    """

    def __init__(self, env):
        super(StateData, self).__init__(env)
        self.current_obs = None

    def reset(self, **kwargs) -> Tuple[ObsType, dict]:
        obs, info = self.env.reset(**kwargs)
        self.current_obs = obs
        return obs, info

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        obs, rew, done, info = super(StateData, self).step(action)
        self.current_obs = obs
        return obs, rew, done, info

    @property
    def state(self):
        if hasattr(self.env, "state"):
            return np.array(self.env.state, dtype=np.float32)
        else:
            return self.current_obs


def wrapping_env(
    env,
    max_episode_steps: Optional[int] = None,
    reward_shift: Optional[float] = None,
    reward_scale: Optional[float] = None,
    obs_shift: Union[np.ndarray, float, list, None] = None,
    obs_scale: Union[np.ndarray, float, list, None] = None,
):
    """Automatically wrap data type environment according to input arguments. Wrapper will not be used
        if all corresponding parameters are set to None.
    :param env: original data type environment.
    :param Optional[int] max_episode_steps: parameter for gym.wrappers.time_limit.TimeLimit wrapper.
        if it is set to None but environment has 'max_episode_steps' attribute, it will be filled in
        TimeLimit wrapper alternatively.
    :param Optional[float] reward_shift: parameter for reward shaping wrapper.
    :param Optional[float] reward_scale: parameter for reward shaping wrapper.
    :param Union[np.ndarray, float, list, None] obs_shift: parameter for observation scale wrapper.
    :param Union[np.ndarray, float, list, None] obs_scale: parameter for observation scale wrapper.
    :param Optional[str] obs_noise_type: parameter for observation noise wrapper.
    :param Optional[list] obs_noise_data: parameter for observation noise wrapper.
    :param Optional[int] repeat_num: parameter for action repeat wrapper.
    :param Optional[bool] sum_reward: parameter for action repeat wrapper.
    :return: wrapped data type environment.
    """
    env = ResetInfoData(env)
    if max_episode_steps is None and hasattr(env, "max_episode_steps"):
        max_episode_steps = getattr(env, "max_episode_steps")
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps)
    env = ConvertType(env)
    env = StateData(env)

    if not all_none(reward_scale, reward_shift):
        reward_scale = 1.0 if reward_scale is None else reward_scale
        reward_shift = 0.0 if reward_shift is None else reward_shift
        env = ShapingRewardData(env, reward_shift, reward_scale)


    if not all_none(obs_shift, obs_scale):
        obs_scale = 1.0 if obs_scale is None else obs_scale
        obs_shift = 0.0 if obs_shift is None else obs_shift
        env = ScaleObservationData(env, obs_shift, obs_scale)

    return env


class ScaleObservationData(gym.Wrapper):
    def __init__(
        self,
        env,
        shift: Union[np.ndarray, float, list] = 0.0,
        scale: Union[np.ndarray, float, list] = 1.0,
    ):
        super(ScaleObservationData, self).__init__(env)
        if isinstance(shift, list):
            shift = np.array(shift, dtype=np.float32)
        if isinstance(scale, list):
            scale = np.array(scale, dtype=np.float32)
        self.shift = shift
        self.scale = scale

    def observation(self, observation):
        return (observation + self.shift) * self.scale

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs_scaled = self.observation(obs)
        info["raw_obs"] = obs
        return obs_scaled, info

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        obs, r, d, info = self.env.step(action)
        obs_scaled = self.observation(obs)
        info["raw_obs"] = obs
        return obs_scaled, r, d, info

class ConvertType(gym.Wrapper):
    """Wrapper converts data type of action and observation to satisfy requirements of original
        environment and gops interface.
    :param env: data type environment.
    """

    def __init__(self, env):
        super(ConvertType, self).__init__(env)
        self.obs_data_tpye = env.observation_space.dtype
        self.act_data_type = env.action_space.dtype
        self.gops_data_type = np.float32

    def reset(self, **kwargs) -> Tuple[ObsType, dict]:
        obs, info = self.env.reset(**kwargs)
        obs = obs.astype(self.gops_data_type)
        return obs, info

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        action = action.astype(self.act_data_type)
        obs, rew, done, info = super(ConvertType, self).step(action)
        obs = obs.astype(self.gops_data_type)
        return obs, rew, done, info
