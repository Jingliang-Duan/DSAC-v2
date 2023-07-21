import gym


def env_creator(**kwargs):
    """
    make env `MountainCarContinuous-v0`
    """
    return gym.make("MountainCarContinuous-v0")
