import gym


def env_creator(**kwargs):
    try:
        return gym.make("SpaceInvaders-v0")
    except:
        raise ModuleNotFoundError("Atari_py is not installed")
