import torch

EPS = 1e-6

class Action_Distribution:
    def __init__(self):
        super().__init__()

    def get_act_dist(self, logits):
        act_dist_cls = getattr(self, "action_distribution_cls")
        has_act_lim = hasattr(self, "act_high_lim")

        act_dist = act_dist_cls(logits)
        if has_act_lim:
            act_dist.act_high_lim = getattr(self, "act_high_lim")
            act_dist.act_low_lim = getattr(self, "act_low_lim")

        return act_dist


class TanhGaussDistribution:
    def __init__(self, logits):
        self.logits = logits
        self.mean, self.std = torch.chunk(logits, chunks=2, dim=-1)
        self.gauss_distribution = torch.distributions.Independent(
            base_distribution=torch.distributions.Normal(self.mean, self.std),
            reinterpreted_batch_ndims=1,
        )
        self.act_high_lim = torch.tensor([1.0])
        self.act_low_lim = torch.tensor([-1.0])

    def sample(self):
        action = self.gauss_distribution.sample()
        action_limited = (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(
            action
        ) + (self.act_high_lim + self.act_low_lim) / 2
        log_prob = (
                self.gauss_distribution.log_prob(action)
                - torch.log(1 + EPS - torch.pow(torch.tanh(action), 2)).sum(-1)
                - torch.log((self.act_high_lim - self.act_low_lim) / 2).sum(-1)
        )
        return action_limited, log_prob

    def rsample(self):
        action = self.gauss_distribution.rsample()
        action_limited = (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(
            action
        ) + (self.act_high_lim + self.act_low_lim) / 2
        log_prob = (
                self.gauss_distribution.log_prob(action)
                - torch.log(1 + EPS - torch.pow(torch.tanh(action), 2)).sum(-1)
                - torch.log((self.act_high_lim - self.act_low_lim) / 2).sum(-1)
        )
        return action_limited, log_prob

    def log_prob(self, action_limited) -> torch.Tensor:
        action = torch.atanh(
            (1 - EPS)
            * (2 * action_limited - (self.act_high_lim + self.act_low_lim))
            / (self.act_high_lim - self.act_low_lim)
        )
        log_prob = self.gauss_distribution.log_prob(action) - torch.log(
            (self.act_high_lim - self.act_low_lim)
            * (1 + EPS - torch.pow(torch.tanh(action), 2))
        ).sum(-1)
        return log_prob

    def entropy(self):
        return self.gauss_distribution.entropy()

    def mode(self):
        return (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(self.mean) + (
                self.act_high_lim + self.act_low_lim
        ) / 2

    def kl_divergence(self, other: "GaussDistribution") -> torch.Tensor:
        return torch.distributions.kl.kl_divergence(
            self.gauss_distribution, other.gauss_distribution
        )


class GaussDistribution:
    def __init__(self, logits):
        self.logits = logits
        self.mean, self.std = torch.chunk(logits, chunks=2, dim=-1)
        self.gauss_distribution = torch.distributions.Independent(
            base_distribution=torch.distributions.Normal(self.mean, self.std),
            reinterpreted_batch_ndims=1,
        )
        self.act_high_lim = torch.tensor([1.0])
        self.act_low_lim = torch.tensor([-1.0])

    def sample(self):
        action = self.gauss_distribution.sample()
        log_prob = self.gauss_distribution.log_prob(action)
        return action, log_prob

    def rsample(self):
        action = self.gauss_distribution.rsample()
        log_prob = self.gauss_distribution.log_prob(action)
        return action, log_prob

    def log_prob(self, action) -> torch.Tensor:
        log_prob = self.gauss_distribution.log_prob(action)
        return log_prob

    def entropy(self):
        return self.gauss_distribution.entropy()

    def mode(self):
        return torch.clamp(self.mean, self.act_low_lim, self.act_high_lim)

    def kl_divergence(self, other: "GaussDistribution") -> torch.Tensor:
        return torch.distributions.kl.kl_divergence(
            self.gauss_distribution, other.gauss_distribution
        )