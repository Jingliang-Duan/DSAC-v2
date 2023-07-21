import gym


def env_creator(**kwargs):
    try:
        return gym.make("Enduro-v0")
    except:
        raise ModuleNotFoundError("Warning: Atari_py is not installed")
