import torch
from .abstractcost import AbstractCost


class UnicycleRegCost(AbstractCost):
    def __init__(self, RQratio):
        self.Q = torch.eye(2)
        self.R = torch.eye(2) * RQratio

    def stage_cost(self, xt, ut, terminal=False):
        """ Quadratic state cost
        :Parameters:
            - xt:           torch.tensor with shape (batch_size, Nx)
            - ut:           torch.tensor with shape (batch_size, Nu)
            - terminal:     bool; whether this is the terminal state
        """
        state_cost = (xt[:, :2].T * (self.Q @ xt[:, :2].T)).sum(0)
        control_cost = (ut.T * (self.R @ ut.T)).sum(0)
        return state_cost + control_cost
