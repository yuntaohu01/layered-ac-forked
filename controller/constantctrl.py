import torch
from .abstractcontroller import AbstractController


class ZeroController(AbstractController):
    """ Generates zeros control actions. (So we get the autonomous dynamics)
    """

    def __init__(self, Nu):
        super().__init__()
        self.Nu = Nu

    def control(self, x):
        return torch.zeros(
            (x.size(0), self.Nu), dtype=torch.double, device=x.device
        )
