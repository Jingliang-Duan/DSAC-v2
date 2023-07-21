#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Mountaincar Environment (continous version)
#  Update Date: 2021-05-55, Yuhang Zhang: create environment


import gym


def env_creator(**kwargs):
    """
    make env `MountainCarContinuous-v0`
    """
    return gym.make("MountainCarContinuous-v0")
