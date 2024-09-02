import torch
from .abstractcontroller import AbstractController


class LQTrackingController(AbstractController):
    """ The controller solves the LQ tracking problem of
            minimize_u  \sum_t (xt-rt) Q (xt-rt) + ut R ut
    """

    def __init__(self, Q, R, A, B, horizon=10):
        self.Q = Q
        self.R = R
        self.A = A
        self.B = B
        self.Nx, self.Nu = self.B.shape
        self.horizon = horizon

    def control(self, x0_refs):
        """ Return the tracking control action
        :Parameters:
            - x0_refs:  torch.Tensor(batch_size, Nx * T). We assume the input
                        comes in the form of (x0, r0, r1, r2, ...).flatten()
        """
        batch_size = x0_refs.size(0)
        u = torch.zeros((batch_size, self.Nu))

        for i, x0_ref in enumerate(x0_refs):
            x0_ref = x0_ref.reshape(-1, self.Nx)
            K, k = self.backward_pass(x0_ref[1:self.horizon])
            u[i] = K @ x0_ref[0] + k
        return u

    def backward_pass(self, x_traj):
        """ for one single trajectory only; not meant to be used in batch """
        T_, Nx = x_traj.shape
        T = T_ - 1

        P, p = self.Q, self.Q @ x_traj[-1]
        for t in range(T-1, -1, -1):
            At, Bt = self.A, self.B
            Qt, Rt, St, qt, rt = self.Q, self.R, 0, self.Q @ x_traj[t], 0
            H_xx = Qt + At.T @ P @ At
            H_uu = Rt + Bt.T @ P @ Bt
            H_xu = St + At.T @ P @ Bt
            h_x = qt + At.T @ p
            h_u = rt + Bt.T @ p
            H_uu_inv = torch.linalg.pinv(H_uu)
            Kt = -H_uu_inv @ H_xu.T
            kt = -H_uu_inv @ h_u
            P = H_xx + H_xu @ Kt
            p = h_x + H_xu @ kt
        # Note that we only need the first control action
        return Kt, kt
