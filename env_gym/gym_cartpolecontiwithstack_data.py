#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Acrobat Environment
#  Update Date: 2021-05-55, Hao Sun: create environment


import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from gym.wrappers.time_limit import TimeLimit

gym.logger.setLevel(gym.logger.ERROR)


class _GymCartpoleconti(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self, **kwargs):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        # Actually half the pole's length
        self.length = 0.5
        self.polemass_length = self.masspole * self.length
        self.force_mag = 30.0
        # Seconds between state updates
        self.tau = 0.02
        self.min_action = -1.0
        self.max_action = 1.0

        # Self.is_adversary = kwargs['is_adversary']

        self.min_adv_action = -0.5
        self.max_adv_action = 0.5

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # Is still within bounds
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ]
        )

        self.action_space = spaces.Box(
            low=self.min_action, high=self.max_action, shape=(1,)
        )
        self.stack_length = 5
        self.observation_space = spaces.Box(
            np.array([-high] * self.stack_length), np.array([high] * self.stack_length)
        )

        self.adv_action_space = spaces.Box(
            low=self.min_adv_action, high=self.max_adv_action, shape=(1,)
        )

        self.seed()
        self.viewer = None
        self.state = None
        self.stack = None

        self.steps_beyond_done = None

        # Self.max_episode_steps = 200
        self.steps = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def stepPhysics(self, force, advu):

        x, x_dot, theta, theta_dot = self.state
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (
            force + self.polemass_length * theta_dot * theta_dot * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length
            * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass)
        )

        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc + advu
        return (x, x_dot, theta, theta_dot)

    def step(self, action, adv_action=None):
        action = np.expand_dims(action, 0)
        if adv_action is None:
            adv_action = 0

        force = self.force_mag * float(action)
        self.state = self.stepPhysics(force, float(adv_action))
        x, x_dot, theta, theta_dot = self.state
        done = (
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        done = bool(done)

        # -----------------
        self.steps += 1
        # If self.steps >=self.max_episode_steps:
        #     done = True
        # ---------------

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                gym.logger.warn(
                    """
You are calling 'step()' even though this environment has already returned
done = True. You should always call 'reset()' once you receive 'done = True'
Any further steps are undefined behavior.
                """
                )
            self.steps_beyond_done += 1
            reward = 0.0

        self.stack.pop(0)
        self.stack.append(self.state)

        return np.array(self.stack), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(
            low=[-2, -0.05, -0.2, -0.05], high=[2, 0.05, 0.2, 0.05], size=(4,)
        )
        self.steps_beyond_done = None
        self.steps = 0
        self.stack = [self.state] * self.stack_length
        return np.array(self.stack)

    def render(self, mode="human"):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        # TOP OF CART
        carty = 100
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = (
                -polewidth / 2,
                polewidth / 2,
                polelen - polewidth / 2,
                -polewidth / 2,
            )
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(0.8, 0.6, 0.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(0.5, 0.5, 0.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self.state is None:
            return None

        x = self.state
        # MIDDLE OF CART
        cartx = x[0] * scale + screen_width / 2.0
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=(mode == "rgb_array"))

    def close(self):
        if self.viewer:
            self.viewer.close()


def env_creator(**kwargs):
    return TimeLimit(_GymCartpoleconti(**kwargs), 200)


if __name__ == "__main__":
    e = env_creator()
    print(e.reset().shape)

    for _ in range(100):
        print(e.step(e.action_space.sample()))
        e.render()
