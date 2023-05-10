#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Noise Function
#  Update Date: 2021-03-10, Yuhang Zhang: Revise Codes


import numpy as np



class EpsilonGreedy:
    def __init__(self, epsilon, action_num):
        self.epsilon = epsilon
        self.action_num = action_num

    def sample(self, action):
        if np.random.random() > self.epsilon:
            return action
        else:
            return np.random.randint(self.action_num)


class GaussNoise:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def sample(self, action):
        return action + np.random.normal(self.mean, self.std)
