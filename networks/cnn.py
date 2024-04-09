#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Convolutional Neural NetworksAction (CNN)
#  Update: 2021-03-05, Wenjun Zou: create CNN function


__all__ = [
    "DetermPolicy",
    "FiniteHorizonPolicy",
    "StochaPolicy",
    "ActionValue",
    "ActionValueDis",
    "StateValue",
    "ActionValueDistri",
]

import torch
import warnings
import torch.nn as nn
from utils.common_utils import get_activation_func
from utils.act_distribution_cls import Action_Distribution


def CNN(kernel_sizes, channels, strides, activation, input_channel):
    """Implementation of CNN.
    :param list kernel_sizes: list of kernel_size,
    :param list channels: list of channels,
    :param list strides: list of stride,
    :param activation: activation function,
    :param int input_channel: number of channels of input image.
    Return CNN.
    Input shape for CNN: (batch_size, channel_num, height, width).
    """
    layers = []
    for j in range(len(kernel_sizes)):
        act = activation
        if j == 0:
            layers += [
                nn.Conv2d(input_channel, channels[j], kernel_sizes[j], strides[j]),
                act(),
            ]
        else:
            layers += [
                nn.Conv2d(channels[j - 1], channels[j], kernel_sizes[j], strides[j]),
                act(),
            ]
    return nn.Sequential(*layers)


# Define MLP function
def MLP(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class DetermPolicy(nn.Module, Action_Distribution):
    """
    Approximated function of deterministic policy.
    Input: observation.
    Output: action.
    """

    def __init__(self, **kwargs):
        super(DetermPolicy, self).__init__()
        act_dim = kwargs["act_dim"]
        obs_dim = kwargs["obs_dim"]
        conv_type = kwargs["conv_type"]
        act_high_lim = kwargs["act_high_lim"]
        act_low_lim = kwargs["act_low_lim"]
        self.register_buffer("act_high_lim", torch.from_numpy(act_high_lim))
        self.register_buffer("act_low_lim", torch.from_numpy(act_low_lim))
        self.hidden_activation = get_activation_func(kwargs["hidden_activation"])
        self.output_activation = get_activation_func(kwargs["output_activation"])
        self.action_distribution_cls = kwargs["action_distribution_cls"]
        if conv_type == "type_1":
            # CNN+MLP Parameters
            conv_kernel_sizes = [8, 4, 3]
            conv_channels = [32, 64, 64]
            conv_strides = [4, 2, 1]
            conv_activation = nn.ReLU
            conv_input_channel = obs_dim[0]
            mlp_hidden_layers = [512, 256]

            # Construct CNN+MLP
            self.conv = CNN(
                conv_kernel_sizes,
                conv_channels,
                conv_strides,
                conv_activation,
                conv_input_channel,
            )
            conv_num_dims = (
                self.conv(torch.ones(obs_dim).unsqueeze(0)).reshape(1, -1).shape[-1]
            )
            mlp_sizes = [conv_num_dims] + mlp_hidden_layers + [act_dim]

            self.mlp = MLP(mlp_sizes, self.hidden_activation, self.output_activation)

        elif conv_type == "type_2":
            # CNN+MLP Parameters
            conv_kernel_sizes = [4, 3, 3, 3, 3, 3]
            conv_channels = [8, 16, 32, 64, 128, 256]
            conv_strides = [2, 2, 2, 2, 1, 1]
            conv_activation = nn.ReLU
            conv_input_channel = obs_dim[0]
            mlp_hidden_layers = [256,256,256]

            # Construct CNN+MLP
            self.conv = CNN(
                conv_kernel_sizes,
                conv_channels,
                conv_strides,
                conv_activation,
                conv_input_channel,
            )
            conv_num_dims = (
                self.conv(torch.ones(obs_dim).unsqueeze(0)).reshape(1, -1).shape[-1]
            )
            # print(conv_num_dims)
            mlp_sizes = [conv_num_dims] + mlp_hidden_layers + [act_dim]

            self.mlp = MLP(mlp_sizes, self.hidden_activation, self.output_activation)
        else:
            raise NotImplementedError

    def forward(self, obs):
        # obs = obs.permute(0, 3, 1, 2)
        img = self.conv(obs)
        feature = img.view(img.size(0), -1)
        feature = self.mlp(feature)
        action = (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(feature) + (
            self.act_high_lim + self.act_low_lim
        ) / 2
        return action


class FiniteHorizonPolicy(nn.Module, Action_Distribution):
    def __init__(self, **kwargs):
        raise NotImplementedError


class StochaPolicy(nn.Module, Action_Distribution):
    """
    Approximated function of stochastic policy.
    Input: observation.
    Output: parameters of action distribution.
    """

    def __init__(self, **kwargs):
        super(StochaPolicy, self).__init__()
        act_dim = kwargs["act_dim"]
        obs_dim = kwargs["obs_dim"]
        conv_type = kwargs["conv_type"]
        act_high_lim = kwargs["act_high_lim"]
        act_low_lim = kwargs["act_low_lim"]
        self.register_buffer("act_high_lim", torch.from_numpy(act_high_lim))
        self.register_buffer("act_low_lim", torch.from_numpy(act_low_lim))
        self.hidden_activation = get_activation_func(kwargs["hidden_activation"])
        self.output_activation = get_activation_func(kwargs["output_activation"])
        self.min_log_std = kwargs["min_log_std"]
        self.max_log_std = kwargs["max_log_std"]
        self.action_distribution_cls = kwargs["action_distribution_cls"]

        if conv_type == "type_1":
            # CNN+MLP Parameters
            conv_kernel_sizes = [8, 4, 3]
            conv_channels = [32, 64, 64]
            conv_strides = [4, 2, 1]
            conv_activation = nn.ReLU
            conv_input_channel = obs_dim[0]
            mlp_hidden_layers = [512, 256]

            # Construct CNN+MLP
            self.conv = CNN(
                conv_kernel_sizes,
                conv_channels,
                conv_strides,
                conv_activation,
                conv_input_channel,
            )
            conv_num_dims = (
                self.conv(torch.ones(obs_dim).unsqueeze(0)).reshape(1, -1).shape[-1]
            )
            policy_mlp_sizes = [conv_num_dims] + mlp_hidden_layers + [act_dim]
            self.mean = MLP(
                policy_mlp_sizes, self.hidden_activation, self.output_activation
            )
            self.log_std = MLP(
                policy_mlp_sizes, self.hidden_activation, self.output_activation
            )

        elif conv_type == "type_2":
            # CNN+MLP Parameters
            conv_kernel_sizes = [4, 3, 3, 3, 3, 3]
            conv_channels = [8, 16, 32, 64, 128, 256]
            conv_strides = [2, 2, 2, 2, 1, 1]
            conv_activation = nn.ReLU
            conv_input_channel = obs_dim[0]
            mlp_hidden_layers = [256,256,256]

            # Construct CNN+MLP
            self.conv = CNN(
                conv_kernel_sizes,
                conv_channels,
                conv_strides,
                conv_activation,
                conv_input_channel,
            )
            conv_num_dims = (
                self.conv(torch.ones(obs_dim).unsqueeze(0)).reshape(1, -1).shape[-1]
            )
            # print(conv_num_dims)
            policy_mlp_sizes = [conv_num_dims] + mlp_hidden_layers + [act_dim]
            self.mean = MLP(
                policy_mlp_sizes, self.hidden_activation, self.output_activation
            )
            self.log_std = MLP(
                policy_mlp_sizes, self.hidden_activation, self.output_activation
            )

        else:
            raise NotImplementedError

    def forward(self, obs):
        img = self.conv(obs)
        feature = img.view(img.size(0), -1)
        action_mean = self.mean(feature)
        action_std = torch.clamp(
            self.log_std(feature), self.min_log_std, self.max_log_std
        ).exp()
        return torch.cat((action_mean, action_std), dim=-1)


class ActionValue(nn.Module, Action_Distribution):
    """
    Approximated function of action-value function.
    Input: observation, action.
    Output: action-value.
    """

    def __init__(self, **kwargs):
        super(ActionValue, self).__init__()
        act_dim = kwargs["act_dim"]
        obs_dim = kwargs["obs_dim"]
        conv_type = kwargs["conv_type"]
        self.hidden_activation = get_activation_func(kwargs["hidden_activation"])
        self.output_activation = get_activation_func(kwargs["output_activation"])
        self.action_distribution_cls = kwargs["action_distribution_cls"]
        if conv_type == "type_1":
            # CNN+MLP Parameters
            conv_kernel_sizes = [8, 4, 3]
            conv_channels = [32, 64, 64]
            conv_strides = [4, 2, 1]
            conv_activation = nn.ReLU
            conv_input_channel = obs_dim[0]
            mlp_hidden_layers = [512, 256]

            # Construct CNN+MLP
            self.conv = CNN(
                conv_kernel_sizes,
                conv_channels,
                conv_strides,
                conv_activation,
                conv_input_channel,
            )
            conv_num_dims = (
                self.conv(torch.ones(obs_dim).unsqueeze(0)).reshape(1, -1).shape[-1]
            )
            mlp_sizes = [conv_num_dims + act_dim] + mlp_hidden_layers + [1]

            self.mlp = MLP(mlp_sizes, self.hidden_activation, self.output_activation)

        elif conv_type == "type_2":
            # CNN+MLP Parameters
            conv_kernel_sizes = [4, 3, 3, 3, 3, 3]
            conv_channels = [8, 16, 32, 64, 128, 256]
            conv_strides = [2, 2, 2, 2, 1, 1]
            conv_activation = nn.ReLU
            conv_input_channel = obs_dim[0]
            mlp_hidden_layers = [256,256,256]

            # Construct CNN+MLP
            self.conv = CNN(
                conv_kernel_sizes,
                conv_channels,
                conv_strides,
                conv_activation,
                conv_input_channel,
            )
            conv_num_dims = (
                self.conv(torch.ones(obs_dim).unsqueeze(0)).reshape(1, -1).shape[-1]
            )
            mlp_sizes = [conv_num_dims + act_dim] + mlp_hidden_layers + [1]
            self.mlp = MLP(mlp_sizes, self.hidden_activation, self.output_activation)
        else:
            raise NotImplementedError

    def forward(self, obs, act):
        img = self.conv(obs)
        feature = torch.cat([img.view(img.size(0), -1), act], -1)
        return self.mlp(feature)


class ActionValueDis(nn.Module, Action_Distribution):
    """
    Approximated function of action-value function for discrete action space.
    Input: observation.
    Output: action-value for all action.
    """

    def __init__(self, **kwargs):
        super(ActionValueDis, self).__init__()
        act_num = kwargs["act_num"]
        obs_dim = kwargs["obs_dim"]
        conv_type = kwargs["conv_type"]
        self.hidden_activation = get_activation_func(kwargs["hidden_activation"])
        self.output_activation = get_activation_func(kwargs["output_activation"])
        self.action_distribution_cls = kwargs["action_distribution_cls"]
        if conv_type == "type_1":
            # CNN+MLP Parameters
            conv_kernel_sizes = [8, 4, 3]
            conv_channels = [32, 64, 64]
            conv_strides = [4, 2, 1]
            conv_activation = nn.ReLU
            conv_input_channel = obs_dim[0]
            mlp_hidden_layers = [512]

            # Construct CNN+MLP
            self.conv = CNN(
                conv_kernel_sizes,
                conv_channels,
                conv_strides,
                conv_activation,
                conv_input_channel,
            )
            conv_num_dims = (
                self.conv(torch.ones(obs_dim).unsqueeze(0)).reshape(1, -1).shape[-1]
            )
            mlp_sizes = [conv_num_dims] + mlp_hidden_layers + [act_num]
            self.mlp = MLP(mlp_sizes, self.hidden_activation, self.output_activation)

        elif conv_type == "type_2":
            # CNN+MLP Parameters
            conv_kernel_sizes = [4, 3, 3, 3, 3, 3]
            conv_channels = [8, 16, 32, 64, 128, 256]
            conv_strides = [2, 2, 2, 2, 1, 1]
            conv_activation = nn.ReLU
            conv_input_channel = obs_dim[0]
            mlp_hidden_layers = [256,256,256]

            # Construct CNN+MLP
            self.conv = CNN(
                conv_kernel_sizes,
                conv_channels,
                conv_strides,
                conv_activation,
                conv_input_channel,
            )
            conv_num_dims = (
                self.conv(torch.ones(obs_dim).unsqueeze(0)).reshape(1, -1).shape[-1]
            )
            mlp_sizes = [conv_num_dims] + mlp_hidden_layers + [act_num]
            self.mlp = MLP(mlp_sizes, self.hidden_activation, self.output_activation)
        else:
            raise NotImplementedError

    def forward(self, obs):
        img = self.conv(obs)
        feature = img.view(img.size(0), -1)
        act_value_dis = self.mlp(feature)
        return torch.squeeze(act_value_dis, -1)


class ActionValueDistri(nn.Module):
    """
    Approximated function of distributed action-value function.
    Input: observation.
    Output: parameters of action-value distribution.
    """

    def __init__(self, **kwargs):
        super(ActionValueDistri, self).__init__()
        act_dim = kwargs["act_dim"]
        obs_dim = kwargs["obs_dim"]
        conv_type = kwargs["conv_type"]
        self.hidden_activation = get_activation_func(kwargs["hidden_activation"])
        self.output_activation = get_activation_func(kwargs["output_activation"])
        self.action_distribution_cls = kwargs["action_distribution_cls"]
        if "min_log_std" in kwargs or "max_log_std" in kwargs:
            warnings.warn("min_log_std and max_log_std are deprecated in ActionValueDistri.")
        if conv_type == "type_1":
            # CNN+MLP Parameters
            conv_kernel_sizes = [8, 4, 3]
            conv_channels = [32, 64, 64]
            conv_strides = [4, 2, 1]
            conv_activation = nn.ReLU
            conv_input_channel = obs_dim[0]
            mlp_hidden_layers = [512, 256]

            # Construct CNN+MLP
            self.conv = CNN(
                conv_kernel_sizes,
                conv_channels,
                conv_strides,
                conv_activation,
                conv_input_channel,
            )
            conv_num_dims = (
                self.conv(torch.ones(obs_dim).unsqueeze(0)).reshape(1, -1).shape[-1]
            )
            mlp_sizes = [conv_num_dims + act_dim] + mlp_hidden_layers + [1]
            self.mean = MLP(mlp_sizes, self.hidden_activation, self.output_activation)
            self.log_std = MLP(
                mlp_sizes, self.hidden_activation, self.output_activation
            )

        elif conv_type == "type_2":
            # CNN+MLP Parameters
            conv_kernel_sizes = [4, 3, 3, 3, 3, 3]
            conv_channels = [8, 16, 32, 64, 128, 256]
            conv_strides = [2, 2, 2, 2, 1, 1]
            conv_activation = nn.ReLU
            conv_input_channel = obs_dim[0]
            mlp_hidden_layers = [256,256,256]

            # Construct CNN+MLP
            self.conv = CNN(
                conv_kernel_sizes,
                conv_channels,
                conv_strides,
                conv_activation,
                conv_input_channel,
            )
            conv_num_dims = (
                self.conv(torch.ones(obs_dim).unsqueeze(0)).reshape(1, -1).shape[-1]
            )
            mlp_sizes = [conv_num_dims + act_dim] + mlp_hidden_layers + [1]
            self.mean = MLP(mlp_sizes, self.hidden_activation, self.output_activation)
            self.log_std = MLP(
                mlp_sizes, self.hidden_activation, self.output_activation
            )
        else:
            raise NotImplementedError

    def forward(self, obs, act):
        img = self.conv(obs)
        feature = torch.cat([img.view(img.size(0), -1), act], -1)
        value_mean = self.mean(feature)
        value_std = self.log_std(feature) # note: std, not log_std
        value_std = torch.nn.functional.softplus(value_std)  # avoid 0

        return torch.cat((value_mean, value_std), dim=-1)


class StochaPolicyDis(ActionValueDis, Action_Distribution):
    """
    Approximated function of stochastic policy for discrete action space.
    Input: observation.
    Output: parameters of action distribution.
    """

    pass


class StateValue(nn.Module, Action_Distribution):
    """
    Approximated function of state-value function.
    Input: observation, action.
    Output: state-value.
    """

    def __init__(self, **kwargs):
        super(StateValue, self).__init__()
        obs_dim = kwargs["obs_dim"]
        conv_type = kwargs["conv_type"]
        self.hidden_activation = get_activation_func(kwargs["hidden_activation"])
        self.output_activation = get_activation_func(kwargs["output_activation"])
        self.action_distribution_cls = kwargs["action_distribution_cls"]
        if conv_type == "type_1":
            # CNN+MLP Parameters
            conv_kernel_sizes = [8, 4, 3]
            conv_channels = [32, 64, 64]
            conv_strides = [4, 2, 1]
            conv_activation = nn.ReLU
            conv_input_channel = obs_dim[0]
            mlp_hidden_layers = [512]

            # Construct CNN+MLP
            self.conv = CNN(
                conv_kernel_sizes,
                conv_channels,
                conv_strides,
                conv_activation,
                conv_input_channel,
            )
            conv_num_dims = (
                self.conv(torch.ones(obs_dim).unsqueeze(0)).reshape(1, -1).shape[-1]
            )
            mlp_sizes = [conv_num_dims] + mlp_hidden_layers + [1]
            self.mlp = MLP(mlp_sizes, self.hidden_activation, self.output_activation)

        elif conv_type == "type_2":
            # CNN+MLP Parameters
            conv_kernel_sizes = [4, 3, 3, 3, 3, 3]
            conv_channels = [8, 16, 32, 64, 128, 256]
            conv_strides = [2, 2, 2, 2, 1, 1]
            conv_activation = nn.ReLU
            conv_input_channel = obs_dim[0]
            mlp_hidden_layers = [256,256,256]

            # Construct CNN+MLP
            self.conv = CNN(
                conv_kernel_sizes,
                conv_channels,
                conv_strides,
                conv_activation,
                conv_input_channel,
            )
            conv_num_dims = (
                self.conv(torch.ones(obs_dim).unsqueeze(0)).reshape(1, -1).shape[-1]
            )
            mlp_sizes = [conv_num_dims] + mlp_hidden_layers + [1]
            self.mlp = MLP(mlp_sizes, self.hidden_activation, self.output_activation)
        else:
            raise NotImplementedError

    def forward(self, obs):
        img = self.conv(obs)
        feature = img.view(img.size(0), -1)
        v = self.mlp(feature)
        v = torch.squeeze(v, -1)
        return v
