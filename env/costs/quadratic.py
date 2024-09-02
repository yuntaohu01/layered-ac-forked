from .abstractcost import AbstractCost


class QuadraticCost(AbstractCost):
    def __init__(self, Q, R, QT=None):
        self.Q = Q
        self.R = R
        if QT is None:
            self.QT = Q
        else:
            self.QT = QT

    def stage_cost(self, xt, ut, terminal=False):
        """ Quadratic state cost
        :Parameters:
            - xt:           torch.tensor with shape (batch_size, Nx)
            - ut:           torch.tensor with shape (batch_size, Nu)
            - terminal:     bool; whether this is the terminal state
        """
        if terminal:
            state_cost = (xt.T * (self.QT @ xt.T)).sum(0)
        else:
            state_cost = (xt.T * (self.Q @ xt.T)).sum(0)
        control_cost = (ut.T * (self.R @ ut.T)).sum(0)
        return state_cost + control_cost
