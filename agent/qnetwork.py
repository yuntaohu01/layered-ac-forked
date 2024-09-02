import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import unicycle_tracking_normalization

class SimpleQNetwork(nn.Module):
    def __init__(self, observation_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(observation_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class QuadraticQNetwork(nn.Module):
    def __init__(self, observation_dim, action_dim, eps=1e-5):
        super().__init__()
        full_dim = observation_dim + action_dim
        self.P_sqrt = nn.Linear(full_dim, full_dim, bias=False)
        # num_params = int(full_dim * (full_dim + 1) / 2)
        # self.full_dim = full_dim
        # self.P_compact = nn.Linear(num_params, 1, bias=False)
        # self.uinds = torch.triu_indices(full_dim, full_dim, offset=1)

    def forward(self, x, a):
        xa = torch.cat([x, a], 1)
        pp = self.P_sqrt(xa)
        return -(pp ** 2).sum(1, keepdim=True)

    def get_P(self):
        """ Return P and p. The value func is x^T P x + p^T x + c """
        p_sqrt = self.P_sqrt.weight.data.clone()
        return p_sqrt.T @ p_sqrt


class UnicycleQNetwork(nn.Module):
    def __init__(self, observation_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(observation_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        #x = unicycle_tracking_normalization(x)
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
