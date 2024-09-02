import torch
from .abstractcost import AbstractCost


class QuadraticTrackingCost(AbstractCost):
    def __init__(self, Q, R, QT=None, tracking_mask=None):
        """
        :Parameters:
            - Q:        torch.Tensor with shape (nominal_Nx, nominal_Nx)
            - R:        torch.Tensor with shape (Nu, Nu)
            - QT:       torch.Tensor with shape (nominal_Nx, nominal_Nx)
        """
        if tracking_mask is None:
            self.nominal_Nx = Q.size(0)
            self.tracking_mask = torch.ones(self.nominal_Nx, dtype=torch.bool)
        else:
            self.nominal_Nx = len(tracking_mask)
            self.tracking_mask = tracking_mask
        self.tracked_Nx = self.tracking_mask.sum()
        self.Q = Q
        self.R = R
        if QT is None:
            self.QT = Q
        else:
            self.QT = QT

    def stage_cost(self, xt, x_next, ut, terminal=False):
        """ Quadratic state cost
        :Parameters:
            - xt:       torch.tensor with shape(batch_size, nominal_Nx * (T+1))
            - x_next:   torch.tensor with shape(batch_size, nominal_Nx * (T+1))
            - ut:           torch.tensor with shape (batch_size, Nu)
            - terminal:     bool; whether this is the terminal state
        """
        # Tracking state now is (x0,r0,r1,...) instead of (x0,r1,r2,...)
        nominal_x_target = xt[:, self.nominal_Nx:self.nominal_Nx + self.tracked_Nx]
        actual_nominal_x = xt[:, :self.nominal_Nx][:, self.tracking_mask]

        x_err = nominal_x_target - actual_nominal_x
        state_cost = (x_err.T * (self.Q @ x_err.T)).sum(0)
        if terminal:
            terminal_x_target = x_next[:, self.nominal_Nx:self.nominal_Nx + self.tracked_Nx]
            actual_terminal_x = x_next[:, :self.nominal_Nx][:, self.tracking_mask]
            terminal_err = terminal_x_target - actual_terminal_x
            state_cost += (terminal_err.T * (self.QT @ terminal_err.T)).sum(0)
        control_cost = (ut.T * (self.R @ ut.T)).sum(0)

        return state_cost + control_cost
