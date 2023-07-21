import gym


def env_creator(**kwargs):
    try:
        return gym.make("LunarLanderContinuous-v2")
    except AttributeError:
        raise ModuleNotFoundError("Box2d is not installed")
