import gym


def env_creator(**kwargs):
    try:
        return gym.make("BipedalWalker-v3")
    except AttributeError:
        raise ModuleNotFoundError("Warning: Box2d is not installed")
