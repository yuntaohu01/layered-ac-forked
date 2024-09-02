import torch
from .abstractcost import AbstractCost


class GoToCircleCost(AbstractCost):
    """ The task is to go to a circle. At every step, the agent is
    penalized for its squared distance to the clostest point on the
    circle plus a quadratic of its control input """

    def __init__(self, radius, R):
        """
        :Parameters:
            - radius:   (float) radius of the circle
            - R:        (np.array Nu x Nu) control penalty
        """
        self.radius = radius
        self.R = R
        # TODO: define a state mask
        print("Warning: unit sphere is computed on all states currently")

    def stage_cost(self, xt, ut, terminal=False):
        """ Quadratic state cost
        :Parameters:
            - xt:           torch.tensor with shape (batch_size, Nx)
            - ut:           torch.tensor with shape (batch_size, Nu)
            - terminal:     bool; whether this is the terminal state
        """
        state_cost = (torch.linalg.norm(xt, dim=1) - self.radius) ** 2
        control_cost = (ut.T * (self.R @ ut.T)).sum(0)
        return state_cost + control_cost
