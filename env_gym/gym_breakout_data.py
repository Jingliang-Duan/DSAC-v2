#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Breakout Environment
#  Update Date: 2021-05-55, Yuhang Zhang: create environment

import gym
import numpy as np
from gym.wrappers.atari_preprocessing import AtariPreprocessing
from gym.wrappers.frame_stack import FrameStack
from gym.wrappers.transform_reward import TransformReward


class FireResetEnv(gym.Wrapper):
    """Take action on reset for environments that are fixed until firing.

    Related discussion: https://github.com/openai/baselines/issues/240

    :param gym.Env env: the environment to wrap.
    """

    def __init__(self, env):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self):
        self.env.reset()
        return self.env.step(1)[0]


class MoveChannel(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        old_obs_shape = self.observation_space.shape
        new_obs_shape = (old_obs_shape[2], old_obs_shape[0], old_obs_shape[1])
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=new_obs_shape, dtype=np.float32,
        )

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class ModifiedFrameStack(FrameStack):
    def __init__(self, env, stack_num):
        super().__init__(env, stack_num)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        return obs[:], reward, done, info

    def reset(self, **kwargs):
        return super().reset(**kwargs)[:]


def sign_reward(origin_reward):
    return np.sign(origin_reward)


def env_creator(**kwargs):
    """
    make env `Breakout-v0` from `Atari`
    """
    try:
        env = gym.make("BreakoutNoFrameskip-v4")
        env = AtariPreprocessing(
            env,
            frame_skip=4,
            grayscale_newaxis=False,
            scale_obs=True,
            terminal_on_life_loss=True,
        )
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = TransformReward(env, sign_reward)
        env = ModifiedFrameStack(env, 4)

        return env
    except:
        raise ModuleNotFoundError("Warning: Atari_py is not installed")
