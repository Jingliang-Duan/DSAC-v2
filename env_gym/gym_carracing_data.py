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


class Env:
    """
    Environment wrapper for CarRacing
    """

    def __init__(self):
        self.env = gym.make("CarRacing-v1")
        self.env.seed(0)
        self.reward_threshold = self.env.spec.reward_threshold
        self.img_stack = 4
        self.action_repeat = 4
        self.action_space = self.env.action_space
        self.action_space.low = np.array([0.0, 0.0, 0.0]).astype(np.float32)
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(4, 96, 96), dtype=np.float64
        )

    def reset(self):
        self.counter = 0
        self.av_r = self.reward_memory()

        self.die = False
        img_rgb = self.env.reset()
        img_gray = self.rgb2gray(img_rgb)
        # Four frames for decision
        self.stack = [img_gray] * self.img_stack
        return np.array(self.stack)

    def step(self, action):
        total_reward = 0
        a = action.copy()
        a[0] = a[0] * 2 - 1
        for i in range(self.action_repeat):
            img_rgb, reward, die, info = self.env.step(a)
            total_reward += reward

            # If no reward recently, end the episode
            done = True if self.av_r(reward) <= -0.1 or die else False
            if done or die:
                done = True
                break
        img_gray = self.rgb2gray(img_rgb)
        self.stack.pop(0)
        self.stack.append(img_gray)
        assert len(self.stack) == self.img_stack
        return np.array(self.stack), total_reward, done, info

    def seed(self, seed=None):
        self.env.seed(seed)

    def render(self, *arg):
        self.env.render(*arg)

    def close(self):
        self.env.close()

    @staticmethod
    def rgb2gray(rgb, norm=True):
        # Rgb image -> gray [0, 1]
        gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
        if norm:
            # Normalize
            gray = gray / 128.0 - 1.0
        return gray

    @staticmethod
    def reward_memory():
        # Record reward for last 100 steps
        count = 0
        length = 100
        history = np.zeros(length)

        def memory(reward):
            nonlocal count
            history[count] = reward
            count = (count + 1) % length
            return np.mean(history)

        return memory


def env_creator(**kwargs):
    """
    make env `CarRacing-v1`, a modified version
    """
    try:
        return Env()
    except:
        raise ModuleNotFoundError("Warning: gym[box2d] is not installed")
