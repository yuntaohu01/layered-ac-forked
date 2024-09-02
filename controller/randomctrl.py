import torch
from .abstractcontroller import AbstractController


class RandomController(AbstractController):
    """ Generates random (normal) control actions
    """

    def __init__(self, Nu, mu=0, std=1):
        super().__init__()
        self.Nu = Nu
        self.mu = mu
        self.std = std

    def control(self, x):
        unit_rand = torch.randn(
            (x.size(0), self.Nu), dtype=torch.double, device=x.device
        )
        return unit_rand * self.std + self.mu
