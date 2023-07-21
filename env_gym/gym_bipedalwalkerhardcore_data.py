#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Bipedalwalker-Hardcore Environment
#  Update Date: 2021-05-55, Yuhang Zhang: create environment

import gym


def env_creator(**kwargs):
    """
    make env `BipedalWalkerHardcore-v3` from `Box2d`
    """
    try:
        return gym.make("BipedalWalkerHardcore-v3")
    except AttributeError:
        raise ModuleNotFoundError("Warning: Box2d is not installed")
