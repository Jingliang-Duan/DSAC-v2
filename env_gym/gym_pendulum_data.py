import gym


def env_creator(**kwargs):
    return gym.make("Pendulum-v1")


if __name__ == "__main__":
    env = env_creator()
    env.reset()
    env.render()
    import time

    time.sleep(100)
