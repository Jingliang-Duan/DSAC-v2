
import gym


def env_creator(**kwargs):
    """
    make env `BipedalWalkerHardcore-v3` from `Box2d`
    """
    try:
        return gym.make("BipedalWalkerHardcore-v3")
    except AttributeError:
        raise ModuleNotFoundError("Warning: Box2d is not installed")
