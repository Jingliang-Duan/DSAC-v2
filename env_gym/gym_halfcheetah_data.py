import gym


def env_creator(**kwargs):
    try:
        return gym.make("HalfCheetah-v3")
    except:
        raise ModuleNotFoundError(
            "Warning:  mujoco, mujoco-py and MSVC are not installed properly"
        )


if __name__ == "__main__":
    env = env_creator()
    env.reset()
    for i in range(100):
        a = env.action_space.sample()
        env.step(a)
        env.render()
