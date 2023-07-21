import gym


def env_creator(**kwargs):
    return gym.make("FrozenLake8x8-v1")
