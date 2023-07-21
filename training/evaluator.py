import numpy as np
import torch

from utils.initialization import create_env
from utils.common_utils import set_seed
from dsac import ApproxContainer



class Evaluator:
    def __init__(self, index=0, **kwargs):
        kwargs.update(
            {"reward_scale": None, "repeat_num": None}
        )  # evaluation don't need to scale reward
        self.env = create_env(**kwargs)
        _, self.env = set_seed(kwargs["trainer"], kwargs["seed"], index + 400, self.env)

        self.networks = ApproxContainer(**kwargs)
        self.render = kwargs["is_render"]
        self.num_eval_episode = kwargs["num_eval_episode"]
        self.action_type = kwargs["action_type"]
        self.policy_func_name = kwargs["policy_func_name"]
        self.save_folder = kwargs["save_folder"]
        self.eval_save = kwargs.get("eval_save", False)

        self.print_time = 0
        self.print_iteration = -1

    def load_state_dict(self, state_dict):
        self.networks.load_state_dict(state_dict)

    def run_an_episode(self, iteration, render=True):
        if self.print_iteration != iteration:
            self.print_iteration = iteration
            self.print_time = 0
        else:
            self.print_time += 1
        obs_list = []
        action_list = []
        reward_list = []
        obs, info = self.env.reset()
        done = 0
        info["TimeLimit.truncated"] = False
        while not (done or info["TimeLimit.truncated"]):
            batch_obs = torch.from_numpy(np.expand_dims(obs, axis=0).astype("float32"))
            logits = self.networks.policy(batch_obs)
            action_distribution = self.networks.create_action_distributions(logits)
            action = action_distribution.mode()
            action = action.detach().numpy()[0]
            next_obs, reward, done, next_info = self.env.step(action)
            obs_list.append(obs)
            action_list.append(action)
            obs = next_obs
            info = next_info
            if "TimeLimit.truncated" not in info.keys():
                info["TimeLimit.truncated"] = False
            # Draw environment animation
            if render:
                self.env.render()
            reward_list.append(reward)
        eval_dict = {
            "reward_list": reward_list,
            "action_list": action_list,
            "obs_list": obs_list,
        }
        if self.eval_save:
            np.save(
                self.save_folder
                + "/evaluator/iter{}_ep{}".format(iteration, self.print_time),
                eval_dict,
            )
        episode_return = sum(reward_list)
        return episode_return

    def run_n_episodes(self, n, iteration):
        episode_return_list = []
        for _ in range(n):
            episode_return_list.append(self.run_an_episode(iteration, self.render))
        return np.mean(episode_return_list)

    def run_evaluation(self, iteration):
        return self.run_n_episodes(self.num_eval_episode, iteration)


def create_evaluator(**kwargs):
    evaluator = Evaluator(**kwargs)
    print("Create evaluator successfully!")
    return evaluator
