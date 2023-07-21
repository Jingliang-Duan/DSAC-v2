import gym


def env_creator(**kwargs):
    return gym.make("FrozenLake-v1")
