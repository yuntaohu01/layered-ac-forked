import torch


class AbstractDynamics:
    def __init__(self):
        self.Nx = None
        self.Nu = None

    def step(self, x, u):
        raise NotImplementedError
