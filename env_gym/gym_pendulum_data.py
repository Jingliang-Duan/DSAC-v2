#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Pendulum Environment
#  Update Date: 2021-05-55, Yuhang Zhang: create environment


import gym


def env_creator(**kwargs):
    return gym.make("Pendulum-v1")


if __name__ == "__main__":
    env = env_creator()
    env.reset()
    env.render()
    import time

    time.sleep(100)
