__all__=["ApproxContainer","DSAC_V1"]
import time
from copy import deepcopy
from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.optim import Adam

from typing import Dict
from utils.tensorboard_setup import tb_tags
from utils.initialization import create_apprfunc
from utils.common_utils import get_apprfunc_dict



class ApproxContainer(torch.nn.Module):
    """Approximate function container for DSAC.

    Contains one policy and one action value.
    """

    def __init__(self, **kwargs):
        super().__init__()
        # create q networks
        q_args = get_apprfunc_dict("value", kwargs["value_func_type"], **kwargs)
        self.q: nn.Module = create_apprfunc(**q_args)
        self.q_target = deepcopy(self.q)

        # create policy network
        policy_args = get_apprfunc_dict("policy", kwargs["policy_func_type"], **kwargs)
        self.policy: nn.Module = create_apprfunc(**policy_args)
        self.policy_target = deepcopy(self.policy)

        # set target network gradients
        for p in self.policy_target.parameters():
            p.requires_grad = False
        for p in self.q_target.parameters():
            p.requires_grad = False

        # create entropy coefficient
        self.log_alpha = nn.Parameter(torch.tensor(1, dtype=torch.float32))

        # create optimizers
        self.q_optimizer = Adam(self.q.parameters(), lr=kwargs["value_learning_rate"])
        self.policy_optimizer = Adam(
            self.policy.parameters(), lr=kwargs["policy_learning_rate"]
        )
        self.alpha_optimizer = Adam([self.log_alpha], lr=kwargs["alpha_learning_rate"])

    def create_action_distributions(self, logits):
        return self.policy.get_act_dist(logits)


class DSAC_V1:
    """DSAC algorithm

    Paper: https://ieeexplore.ieee.org/abstract/document/9448360

    :param float gamma: discount factor.
    :param float tau: param for soft update of target network.
    :param bool auto_alpha: whether to adjust temperature automatically.
    :param float alpha: initial temperature.
    :param float TD_bound: the bound of temporal difference.
    :param bool bound: whether to bound the q value.
    :param float delay_update: delay update steps for actor.
    :param Optional[float] target_entropy: target entropy for automatic
        temperature adjustment.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.networks = ApproxContainer(**kwargs)
        self.gamma = kwargs["gamma"]
        self.tau = kwargs["tau"]
        self.target_entropy = -kwargs["action_dim"]
        self.auto_alpha = kwargs["auto_alpha"]
        self.alpha = kwargs.get("alpha", 0.2)
        self.TD_bound = kwargs.get("TD_bound", 20)
        self.bound = kwargs.get("bound",True)
        self.delay_update = kwargs["delay_update"]

    @property
    def adjustable_parameters(self):
        return (
            "gamma",
            "tau",
            "auto_alpha",
            "alpha",
            "TD_bound",
            "bound",
            "delay_update",
        )

    def local_update(self, data: Dict, iteration: int) -> dict:
        tb_info = self.__compute_gradient(data, iteration)
        self.__update(iteration)
        return tb_info

    def get_remote_update_info(
        self, data: Dict, iteration: int
    ) -> Tuple[dict, dict]:
        tb_info = self.__compute_gradient(data, iteration)

        update_info = {
            "q_grad": [p._grad for p in self.networks.q.parameters()],
            "policy_grad": [p._grad for p in self.networks.policy.parameters()],
            "iteration": iteration,
        }
        if self.auto_alpha:
            update_info.update({"log_alpha_grad":self.networks.log_alpha.grad})

        return tb_info, update_info

    def remote_update(self, update_info: dict):
        iteration = update_info["iteration"]
        q_grad = update_info["q_grad"]
        policy_grad = update_info["policy_grad"]

        for p, grad in zip(self.networks.q.parameters(), q_grad):
            p._grad = grad
        for p, grad in zip(self.networks.policy.parameters(), policy_grad):
            p._grad = grad
        if self.auto_alpha:
            self.networks.log_alpha._grad = update_info["log_alpha_grad"]

        self.__update(iteration)

    def __get_alpha(self, requires_grad: bool = False):
        if self.auto_alpha:
            alpha = self.networks.log_alpha.exp()
            if requires_grad:
                return alpha
            else:
                return alpha.item()
        else:
            return self.alpha

    def __compute_gradient(self, data: Dict, iteration: int):
        start_time = time.time()

        obs = data["obs"]
        logits = self.networks.policy(obs)
        policy_mean = torch.tanh(logits[..., 0]).mean().item()
        policy_std = logits[..., 1].mean().item()

        act_dist = self.networks.create_action_distributions(logits)
        new_act, new_log_prob = act_dist.rsample()
        data.update({"new_act": new_act, "new_log_prob": new_log_prob})

        self.networks.q_optimizer.zero_grad()
        loss_q, q, std = self.__compute_loss_q(data)
        loss_q.backward()

        for p in self.networks.q.parameters():
            p.requires_grad = False

        self.networks.policy_optimizer.zero_grad()
        loss_policy, entropy = self.__compute_loss_policy(data)
        loss_policy.backward()

        for p in self.networks.q.parameters():
            p.requires_grad = True

        if self.auto_alpha:
            self.networks.alpha_optimizer.zero_grad()
            loss_alpha = self.__compute_loss_alpha(data)
            loss_alpha.backward()

        tb_info = {
            "DSAC/critic_avg_q-RL iter": q.item(),
            "DSAC/critic_avg_std-RL iter": std.item(),
            tb_tags["loss_actor"]: loss_policy.item(),
            "DSAC/policy_mean-RL iter": policy_mean,
            "DSAC/policy_std-RL iter": policy_std,
            "DSAC/entropy-RL iter": entropy.item(),
            "DSAC/alpha-RL iter": self.__get_alpha(),
            tb_tags["alg_time"]: (time.time() - start_time) * 1000,
        }

        return tb_info

    def __q_evaluate(self, obs, act, qnet):
        StochaQ = qnet(obs, act)
        mean, std = StochaQ[..., 0], StochaQ[..., -1]
        # std = log_std.exp()
        normal = Normal(torch.zeros_like(mean), torch.ones_like(std))
        z = normal.sample()
        z = torch.clamp(z, -3, 3)
        q_value = mean + torch.mul(z, std)
        return mean, std, q_value

    def __compute_loss_q(self, data: Dict):
        obs, act, rew, obs2, done = (
            data["obs"],
            data["act"],
            data["rew"],
            data["obs2"],
            data["done"],
        )
        logits_2 = self.networks.policy_target(obs2)
        act2_dist = self.networks.create_action_distributions(logits_2)
        act2, log_prob_act2 = act2_dist.rsample()

        q, q_std, q_sample = self.__q_evaluate(obs, act, self.networks.q)
        _, _, q_next_sample = self.__q_evaluate(
            obs2, act2, self.networks.q_target
        )
        target_q, target_q_bound = self.__compute_target_q(
            rew,
            done,
            q.detach(),
            q_next_sample.detach(),
            log_prob_act2.detach(),
        )
        if self.bound:
            q_std_detach = torch.clamp(q_std, min=0.).detach()
            bias = 0.1
            q_loss = torch.mean(
                -(target_q - q).detach() / ( torch.pow(q_std_detach, 2)+ bias)*q
                -((torch.pow(q.detach() - target_q_bound, 2)- q_std_detach.pow(2) )/ (torch.pow(q_std_detach, 3) +bias)
                )*q_std
            )
        else:
            q_loss = -Normal(q, q_std).log_prob(target_q).mean()
        return q_loss, q.detach().mean(), q_std.detach().mean()

    def __compute_target_q(self, r, done, q, q_next, log_prob_a_next):
        target_q = r + (1 - done) * self.gamma * (
            q_next - self.__get_alpha() * log_prob_a_next
        )
        difference = torch.clamp(target_q - q, -self.TD_bound, self.TD_bound)
        target_q_bound = q + difference
        return target_q.detach(), target_q_bound.detach()

    def __compute_loss_policy(self, data: Dict):
        obs, new_act, new_log_prob = data["obs"], data["new_act"], data["new_log_prob"]
        q, _, _ = self.__q_evaluate(obs, new_act, self.networks.q)
        loss_policy = (self.__get_alpha() * new_log_prob - q).mean()
        entropy = -new_log_prob.detach().mean()
        return loss_policy, entropy

    def __compute_loss_alpha(self, data: Dict):
        new_log_prob = data["new_log_prob"]
        loss_alpha = (
            -self.networks.log_alpha
            * (new_log_prob.detach() + self.target_entropy).mean()
        )
        return loss_alpha

    def __update(self, iteration: int):
        self.networks.q_optimizer.step()

        if iteration % self.delay_update == 0:
            self.networks.policy_optimizer.step()

            if self.auto_alpha:
                self.networks.alpha_optimizer.step()

            with torch.no_grad():
                polyak = 1 - self.tau
                for p, p_targ in zip(
                    self.networks.q.parameters(), self.networks.q_target.parameters()
                ):
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)
                for p, p_targ in zip(
                    self.networks.policy.parameters(),
                    self.networks.policy_target.parameters(),
                ):
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)
