import gym


def env_creator(**kwargs):
    return gym.make("Taxi-v3")
