#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Acrobat Environment
#  Update: 2021-05-55, Yuhang Zhang: create environment


import gym


def env_creator(**kwargs):
    return gym.make("Acrobot-v1")
