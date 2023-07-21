import gym

def env_creator(**kwargs):
    """
    make env `LunarLander-v2` from `Box2d`
    """
    try:
        return gym.make("LunarLander-v2")
    except AttributeError:
        raise ModuleNotFoundError("Box2d is not installed")
