import time

import numpy as np
import torch

from utils.initialization import create_env
from utils.common_utils import set_seed
from utils.explore_noise import GaussNoise, EpsilonGreedy
from utils.tensorboard_setup import tb_tags
from dsac import ApproxContainer


class OffSampler:
    def __init__(self, index=0, **kwargs):
        # initialize necessary hyperparameters
        self.env = create_env(**kwargs)
        _, self.env = set_seed(kwargs["trainer"], kwargs["seed"], index + 200, self.env)
        self.obs, self.info = self.env.reset()
        self.has_render = hasattr(self.env, "render")
        self.networks = ApproxContainer(**kwargs)
        self.noise_params = kwargs["noise_params"]
        self.sample_batch_size = kwargs["batch_size_per_sampler"]
        self.policy_func_name = kwargs["policy_func_name"]
        self.action_type = kwargs["action_type"]
        self.obsv_dim = kwargs["obsv_dim"]
        self.act_dim = kwargs["action_dim"]
        self.total_sample_number = 0
        self.reward_scale = 1.0
        if self.noise_params is not None:
            if self.action_type == "continu":
                self.noise_processor = GaussNoise(**self.noise_params)
            elif self.action_type == "discret":
                self.noise_processor = EpsilonGreedy(**self.noise_params)

    def load_state_dict(self, state_dict):
        self.networks.load_state_dict(state_dict)

    def sample(self):
        self.total_sample_number += self.sample_batch_size
        tb_info = dict()
        start_time = time.perf_counter()
        batch_data = []
        for _ in range(self.sample_batch_size):
            # take action using behavior policy
            batch_obs = torch.from_numpy(
                np.expand_dims(self.obs, axis=0).astype("float32")
            )
            logits = self.networks.policy(batch_obs)

            action_distribution = self.networks.create_action_distributions(logits)
            action, logp = action_distribution.sample()
            action = action.detach()[0].numpy()
            logp = logp.detach()[0].numpy()

            if self.noise_params is not None:
                action = self.noise_processor.sample(action)
            # ensure action is array
            action = np.array(action)
            if self.action_type == "continu":
                action_clip = action.clip(
                    self.env.action_space.low, self.env.action_space.high
                )
            else:
                action_clip = action
            # interact with environment
            next_obs, reward, self.done, next_info = self.env.step(action_clip)
            if "TimeLimit.truncated" not in next_info.keys():
                next_info["TimeLimit.truncated"] = False
            if next_info["TimeLimit.truncated"]:
                self.done = False
            data = [
                self.obs.copy(),
                self.info,
                action,
                self.reward_scale * reward,
                next_obs.copy(),
                self.done,
                logp,
                next_info,
            ]
            batch_data.append(tuple(data))
            self.obs = next_obs
            self.info = next_info
            if self.done or next_info["TimeLimit.truncated"]:
                self.obs, self.info = self.env.reset()

        end_time = time.perf_counter()
        tb_info[tb_tags["sampler_time"]] = (end_time - start_time) * 1000

        return batch_data, tb_info

    def get_total_sample_number(self):
        return self.total_sample_number


def create_sampler(**kwargs):
    sampler = OffSampler(**kwargs)
    return sampler
