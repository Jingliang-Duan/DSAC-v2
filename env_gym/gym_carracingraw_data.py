#  Copyright (c). All Rights Reserved.
#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Car-racing Environment
#  Update Date: 2021-05-55, Yuhang Zhang: create environment


import gym
from gym.utils import seeding
import numpy as np


class Env(gym.Env):
    """
    Environment wrapper for CarRacing
    """

    def __init__(self, **kwargs):
        self.env = gym.make("CarRacing-v1")
        # self.env.seed(0)
        self.reward_threshold = self.env.spec.reward_threshold
        self.action_repeat = 4
        self.action_space = self.env.action_space
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(3, 96, 96), dtype=np.float64
        )

    def reset(self):

        img_rgb = self.env.reset()
        img_rgb = self.preprocess(img_rgb)
        return img_rgb

    def step(self, action):
        total_reward = 0
        a = action.copy()
        for i in range(self.action_repeat):
            img_rgb, reward, die, info = self.env.step(a)
            total_reward += reward

            # If no reward recently, end the episode
            done = True if die else False
            if done or die:
                done = True
                break
        img_rgb = self.preprocess(img_rgb)

        return img_rgb, total_reward, done, info

    def seed(self, seed=None):
        self.env.seed(seed)

    def render(self, *arg):
        return self.env.render(*arg)

    def close(self):
        self.env.close()

    @staticmethod
    def preprocess(rgb):
        rgb = rgb.transpose((2,0,1))
        rgb = rgb/255
        return rgb


def env_creator(**kwargs):
    """
    make env `CarRacing-v1`, a modified version
    """

    return Env(**kwargs)

