#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: lunarlander Environment
#  Update Date: 2021-05-55, Yuhang Zhang: create environment


import gym


def env_creator(**kwargs):
    """
    make env `LunarLander-v2` from `Box2d`
    """
    try:
        return gym.make("LunarLander-v2")
    except AttributeError:
        raise ModuleNotFoundError("Box2d is not installed")
