import torch


class ILQRHelper:
    def __init__(self, env):
        """
        Parameters:
            - env:          GymEnv instance
        Synthesizes linear dynamics and quadratic costs from autodiff.
        Specifically,
                linearize_dynamics: (x, u) -> (A, B)
                quadratize_cost:    (x, u) -> (Q, R, S, q, r)
        """
        self.env = env
        self.step = env.dynamics.step
        self.cost = env.cost_fn.stage_cost

        def linearized_dynamics(x, u):
            jacs = torch.autograd.functional.jacobian(
                    lambda x, u: self.step(x, u), (x, u)
            )
            A, B = [j[0, :, 0] for j in jacs]
            return A, B

        def quadratize_cost(x, u):
            jacs = torch.autograd.functional.jacobian(self.cost, (x, u))
            q, r = [j[0, 0] for j in jacs]
            hess = torch.autograd.functional.hessian(self.cost, (x, u))
            Q = hess[0][0][0,:,0]
            R = hess[1][1][0,:,0]
            S = hess[0][1][0,:,0]
            return Q, R, S, q, r

        self.linearized_dynamics = linearized_dynamics
        self.quadratize_cost = quadratize_cost

    def backward_pass(self, x_traj, u_traj):
        """ for one single trajectory only; not meant to be used in batch """
        # TODO: make this into a batch thing
        _, Nx = x_traj.shape
        T, Nu = u_traj.shape
        K, k = [], []

        P, _, _, p, _ = self.quadratize_cost(
            x_traj[-1].unsqueeze(0), torch.zeros_like(u_traj[-1].unsqueeze(0)))
        for t in range(T-1, -1, -1):
            xt, ut = x_traj[t].unsqueeze(0), u_traj[t].unsqueeze(0)
            At, Bt = self.linearized_dynamics(xt, ut)
            Qt, Rt, St, qt, rt = self.quadratize_cost(xt, ut)
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
            K.append(Kt)
            k.append(kt)
        K, k = K[::-1], k[::-1]
        return K, k

    def forward_pass(self, x_old, u_old, K, k):
        T = u_old.shape[0]
        x_new = torch.zeros_like(x_old)
        u_new = torch.zeros_like(u_old)
        x_new[0] = x_old[0]
        for t in range(T):
            u_new[t] = u_old[t] + K[t] @ (x_new[t] - x_old[t]) + k[t]
            x_new[t+1] = self.step(x_new[t].unsqueeze(0), u_new[t].unsqueeze(0))[0]
        return x_new, u_new

    def solve(self, x0=None, x_traj=None, u_traj=None, max_iter=15, thres=1e-3):
        if (x_traj is None) or (u_traj is None):
            x_old, u_old = self.env.sample_random_trajectory(x0=x0, std=0.1)
            x_old, u_old = torch.tensor(x_old), torch.tensor(u_old)
        else:
            x_old, u_old = x_traj, u_traj
        for iter in range(max_iter):
            K, k = self.backward_pass(x_old, u_old)
            x_new, u_new = self.forward_pass(x_old, u_old, K, k)
            if torch.linalg.norm(x_old - x_new) < thres and \
                    torch.linalg.norm(u_old - u_new) < thres:
                break
            x_old, u_old = x_new, u_new
        return x_new, u_new
