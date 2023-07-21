import gym


def env_creator(**kwargs):
    return gym.make("CartPole-v0")
