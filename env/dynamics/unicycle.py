import torch
from .abstractdynamics import AbstractDynamics


class UnicycleDynamics(AbstractDynamics):
    """ Discrete time unicycle dynamics with the 4D state and 2D input
        x = [x, y, velocity, heading angle]
        u = [acceleration, angular velocity] """

    def __init__(self, dt=1e-2):
        """ Initialize the environment. """
        self.Nx = 4
        self.Nu = 2
        self.dt = dt

    def step(self, x, u):
        """ Step the dynamics forward
        :Parameters:
            - x: torch.Tensor of shape (batch, Nx)
            - u: torch.Tensor of shape (batch, Nu)
        """
        dx = self.dt * torch.hstack([
            x[:, 2:3] * torch.cos(x[:, 3:4]),
            x[:, 2:3] * torch.sin(x[:, 3:4]),
            u[:, 0:1],
            u[:, 1:2]])
        return x + dx
