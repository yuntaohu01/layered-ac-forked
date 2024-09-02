import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import unicycle_tracking_normalization

class SimpleActor(nn.Module):
    def __init__(self, observation_dim, action_dim, u_max, u_min):
        super().__init__()
        self.fc1 = nn.Linear(observation_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, action_dim)
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((u_max - u_min) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((u_max + u_min) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


class LinearActor(nn.Module):
    def __init__(self, observation_dim, action_dim):
        super().__init__()
        self.gain = nn.Linear(observation_dim, action_dim, bias=False)
        self.true_gain = None
        self.register_buffer(
            "action_scale", torch.tensor(1., dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor(0., dtype=torch.float32)
        )

    def forward(self, x):
        #true_u = x @ self.true_gain.T
        u = self.gain(x)
        return u


class UnicycleActor(nn.Module):
    """ Learns a residual to the given nominal controller """
    def __init__(self, nominal_controller, observation_dim, action_dim, u_max, u_min):
        super().__init__()
        self.nominal_controller = nominal_controller

        # TODO: specify how many nominal state look aheads to use as a param
        self.actual_obs_dim = 16

        self.fc1 = nn.Linear(self.actual_obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, action_dim)
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((u_max - u_min) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((u_max + u_min) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        #nominal_u = self.nominal_controller.control(x)
        x = x[:, :self.actual_obs_dim]
        #x = unicycle_tracking_normalization(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))

        # Add nominal control with residual
        return x * self.action_scale + self.action_bias #+ nominal_u

