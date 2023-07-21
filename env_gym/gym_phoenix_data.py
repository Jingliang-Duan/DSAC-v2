import gym


def env_creator(**kwargs):
    try:
        return gym.make("Phoenix-v0")
    except:
        raise ModuleNotFoundError("Atari_py not install properly")
