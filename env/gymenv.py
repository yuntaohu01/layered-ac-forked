import torch
import numpy as np

import gymnasium as gym
from gymnasium import spaces


class GymEnv(gym.Env):
    def __init__(self,
                 dynamics,
                 cost_fn,
                 reset_fn,
                 T,
                 xlim=[-np.inf, np.inf],
                 ulim=[-np.inf, np.inf],
                 dtype=np.float32):
        """
        :Parameters:
            - dynamics:     (Dynamics object)
            - cost_fn:      (Cost object)
            - reset_fn:     (Callable: None -> Observation space)
            - T:            (Int) length of the trajectory
            - dtype:        (type) floating point precision to be used
        """
        self.dynamics = dynamics
        self.cost_fn = cost_fn
        self.reset_fn = reset_fn
        self.T = T

        self.action_space = spaces.Box(
            low=ulim[0], high=ulim[1], shape=(self.dynamics.Nu,), dtype=dtype
        )
        self.observation_space = spaces.Box(
            low=xlim[0], high=xlim[1], shape=(self.dynamics.Nx,), dtype=dtype
        )

        self.dtype = dtype

        # We keep self.x to be a batched torch.Tensor
        self.x = None
        self.t = None

    def reset(self, seed=None, x0=None, options=None):
        """ Resets the state back to initial state """
        # Set random seed
        if x0 is not None:
            self.x = torch.tensor(x0).float().unsqueeze(0)
        else:
            if seed:
                np.random.seed(seed)
                torch.manual_seed(seed)
            self.x = self.reset_fn().detach().unsqueeze(0)
        self.t = 0
        return self.x[0].numpy(), {}

    def step(self, u):
        """ step the internal state forward """
        u_tensor = torch.tensor(u).unsqueeze(0).float()
        with torch.no_grad():
            x_ = self.dynamics.step(self.x, u_tensor)
            cost = self.cost_fn.stage_cost(
                self.x, u_tensor, terminal=(self.t == self.T-1)
            )
            reward = -cost
        self.x = x_
        self.t += 1
        return self.x[0].numpy(), reward.item(), self.t >= self.T, False, {}

    def sample_random_trajectory(self, std, x0=None, seed=None):
        """ Generate a (state, control) trajectory of the system by applying
        control sampled from a zero-mean random Gaussian.
        :Parameters:
            - std:          (float) standard deviation of the control input
            - x0:           (np.array) initial state (unbatched)
        :Returns:
            - xtraj:        (np.ndarray (T+1) x Nx) state trajectory
            - utraj:        (np.ndarray (T+1) x Nu) control trajectory
        """
        xtraj, utraj = [], []
        if x0 is None:
            x, _ = self.reset(seed=seed)
        else:
            x, _ = self.reset(x0=x0)
        xtraj.append(x)
        for t in range(self.T):
            u = np.random.randn(self.dynamics.Nu) * std
            x_, r, done, _, _ = self.step(u)
            xtraj.append(x_)
            utraj.append(u)
        return np.vstack(xtraj), np.vstack(utraj).astype(np.float32)
