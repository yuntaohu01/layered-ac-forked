import torch
import torch.autograd.functional as autograd_func
import matplotlib.pyplot as plt

class ILQRtrackingcontroller:
    def __init__(self, dynamics, Q, R, Qf, dt=1.0, device='cpu'):
        """
        Initialize the ILQR Helper.

        Parameters:
            - env: GymEnv instance, must have env.dynamics.step(x, u) method
            - Q: State cost weighting matrix
            - R: Control cost weighting matrix
            - Qf: Terminal state cost weighting matrix
            - dt: Time step
            - device: 'cpu' or 'cuda' for GPU computation
        """
        self.step = dynamics.step
        self.Q = Q.to(device)
        self.R = R.to(device)
        self.Qf = Qf.to(device)
        self.dt = dt
        self.device = device

    def linearized_dynamics(self, x, u):
        """
        Linearize the dynamics function for a batch of states and inputs.
        
        Args:
            step_func (callable): The dynamics function f(x, u) returning x_next.
            x (torch.Tensor): Current state tensor of shape (batch_size, n_x).
            u (torch.Tensor): Current input tensor of shape (batch_size, n_u).
            
        Returns:
            A (torch.Tensor): Jacobian matrices df/dx of shape (batch_size, n_x, n_x).
            B (torch.Tensor): Jacobian matrices df/du of shape (batch_size, n_x, n_u).
        """
        batch_size, n_x = x.shape
        _, n_u = u.shape
        
        # Ensure gradients are tracked
        x = x.clone().detach().requires_grad_(True)
        u = u.clone().detach().requires_grad_(True)
        
        # Compute next states
        x_next = self.step(x, u)  # Shape: (batch_size, n_x)
        n_x_next = x_next.shape[1]
        
        # Initialize Jacobians
        A = torch.zeros(batch_size, n_x_next, n_x)
        B = torch.zeros(batch_size, n_x_next, n_u)
        
        for i in range(n_x_next):
            # Create a tensor of ones for the i-th output
            grad_output = torch.zeros_like(x_next)
            grad_output[:, i] = 1.0
            
            # Compute gradients df_i/dx
            grads_A = torch.autograd.grad(
                outputs=x_next,
                inputs=x,
                grad_outputs=grad_output,
                retain_graph=True,
                create_graph=False,
                allow_unused=True
            )[0]  # Shape: (batch_size, n_x)
            if grads_A is not None:
                A[:, i, :] = grads_A
            
            # Compute gradients df_i/du
            grads_B = torch.autograd.grad(
                outputs=x_next,
                inputs=u,
                grad_outputs=grad_output,
                retain_graph=True,
                create_graph=False,
                allow_unused=True
            )[0]  # Shape: (batch_size, n_u)
            if grads_B is not None:
                B[:, i, :] = grads_B
        return A.to(self.device), B.to(self.device)

    def quadratize_cost(self, x, u, x_ref, u_ref):
        """
        Quadraticize the cost function around the current state and control.

        Parameters:
            - x: Current state, tensor of shape (batch_size, Nx)
            - u: Current control, tensor of shape (batch_size, Nu)
            - x_ref: Reference state, tensor of shape (batch_size, Nx)
            - u_ref: Reference control, tensor of shape (batch_size, Nu)

        Returns:
            - Q: State quadratic term, tensor of shape (batch_size, Nx, Nx)
            - R: Control quadratic term, tensor of shape (batch_size, Nu, Nu)
            - S: State-control quadratic term, tensor of shape (batch_size, Nx, Nu)
            - q: State linear term, tensor of shape (batch_size, Nx)
            - r: Control linear term, tensor of shape (batch_size, Nu)
            - c: Cost constant term, tensor of shape (batch_size,)
        """
        # Compute cost
        dx = x - x_ref  # Shape: (batch_size, Nx)
        du = u - u_ref  # Shape: (batch_size, Nu)
        cost = torch.sum((dx @ self.Q) * dx, dim=1) + torch.sum((du @ self.R) * du, dim=1)  # Shape: (batch_size,)

        # Compute gradients
        l = cost

        # Gradients w.r. x and u
        dL_dx = 2 * dx @ self.Q  # Shape: (batch_size, Nx)
        dL_du = 2 * du @ self.R  # Shape: (batch_size, Nu)

        # Hessians
        Hxx = 2 * self.Q.unsqueeze(0).expand(x.size(0), -1, -1)  # Shape: (batch_size, Nx, Nx)
        Huu = 2 * self.R.unsqueeze(0).expand(u.size(0), -1, -1)  # Shape: (batch_size, Nu, Nu)
        Hxu = torch.zeros(x.size(0), x.size(1), u.size(1), device=self.device)  # No cross terms in this cost

        Q = Hxx
        R = Huu
        S = Hxu

        q = dL_dx
        r = dL_du
        c = l

        return Q, R, S, q, r, c

    def solve(self, x0, u0, x_ref_traj, u_ref_traj, max_iters=100, tol=1e-2, alpha=0.1):
        """
        Execute iLQR to track the reference trajectory.

        Parameters:
            - x0: Initial state, tensor of shape (batch_size, Nx)
            - u0: Initial control sequence, tensor of shape (batch_size, T, Nu)
            - x_ref_traj: Reference state trajectory, tensor of shape (batch_size, T+1, Nx)
            - u_ref_traj: Reference control trajectory, tensor of shape (batch_size, T, Nu)
            - max_iters: Maximum number of iterations
            - tol: Convergence threshold
            - alpha: Step size factor for line search

        Returns:
            - u_traj: Optimized control sequence, tensor of shape (batch_size, T, Nu)
            - x_traj: Optimized state trajectory, tensor of shape (batch_size, T+1, Nx)
        """
        device = self.device
        batch_size, T, Nu = u0.size()
        _, Nx = x0.size()

        # Initialize control and state sequences
        u_traj = u0.clone().detach().to(device)  # Shape: (batch_size, T, Nu)
        x_traj = torch.zeros(batch_size, T+1, Nx, device=device)
        x_traj[:, 0, :] = x0.to(device)

        # Forward simulate initial trajectory
        for t in range(T):
            x_traj[:, t+1, :] = self.step(x_traj[:, t, :], u_traj[:, t, :]).detach()

        prev_cost = torch.zeros(batch_size, device=device)
        for iteration in range(max_iters):
            # Initialize value function at terminal state
            V_x = 2 * (x_traj[:, -1, :] - x_ref_traj[:, -1, :]) @ self.Qf  # Shape: (batch_size, Nx)
            V_xx = 2 * self.Qf.unsqueeze(0).expand(batch_size, -1, -1).clone()  # Shape: (batch_size, Nx, Nx)

            # Initialize feedback and feedforward gains
            K_traj = torch.zeros(batch_size, T, Nu, Nx, device=device)  # Shape: (batch_size, T, Nu, Nx)
            k_traj = torch.zeros(batch_size, T, Nu, device=device)  # Shape: (batch_size, T, Nu)

            # Backward pass
            for t in reversed(range(T)):
                x = x_traj[:, t, :]  # Shape: (batch_size, Nx)
                u = u_traj[:, t, :]  # Shape: (batch_size, Nu)
                x_ref = x_ref_traj[:, t, :]  # Shape: (batch_size, Nx)
                u_ref = u_ref_traj[:, t, :]  # Shape: (batch_size, Nu)

                # Linearize dynamics
                A, B = self.linearized_dynamics(x, u)  # Shapes: (batch_size, Nx, Nx), (batch_size, Nx, Nu)

                # Quadraticize cost
                Q_cost, R_cost, S_cost, q_cost, r_cost, _ = self.quadratize_cost(x, u, x_ref, u_ref)
                # Compute Q-function derivatives
                Q_x = q_cost + torch.bmm(A.transpose(1, 2), V_x.unsqueeze(2)).squeeze(2)  # Shape: (batch_size, Nx)
                Q_u = r_cost + torch.bmm(B.transpose(1, 2), V_x.unsqueeze(2)).squeeze(2)  # Shape: (batch_size, Nu)

                Q_xx = Q_cost + torch.bmm(A.transpose(1, 2), torch.bmm(V_xx, A))  # Shape: (batch_size, Nx, Nx)
                Q_ux = torch.bmm(B.transpose(1, 2), torch.bmm(V_xx, A))  # Shape: (batch_size, Nu, Nx)
                Q_uu = R_cost + torch.bmm(B.transpose(1, 2), torch.bmm(V_xx, B))  # Shape: (batch_size, Nu, Nu)

                # Regularization for numerical stability
                reg = 1e-6 * torch.eye(Nu, device=device).unsqueeze(0).expand(batch_size, -1, -1)
                Q_uu += reg
                # Compute gains
                try:
                    Q_uu_inv = torch.inverse(Q_uu)  # Shape: (batch_size, Nu, Nu)
                except RuntimeError:
                    print("Q_uu is not invertible")
                    return u_traj, x_traj

                K = -torch.bmm(Q_uu_inv, Q_ux)  # Shape: (batch_size, Nu, Nx)
                k = -torch.bmm(Q_uu_inv, Q_u.unsqueeze(2)).squeeze(2)  # Shape: (batch_size, Nu)
                # Update value function
                # V_x = Q_x - torch.bmm(K.transpose(1, 2), torch.bmm(Q_uu, k.unsqueeze(2))).squeeze(2)  # Shape: (batch_size, Nx)
                # V_xx = Q_xx - torch.bmm(K.transpose(1, 2), torch.bmm(Q_uu, K))  # Shape: (batch_size, Nx, Nx)
                V_x = Q_x + torch.bmm(K.transpose(1, 2), Q_u.unsqueeze(2)).squeeze(2)  # Shape: (batch_size, state_dim)
                V_xx = Q_xx + torch.bmm(K.transpose(1, 2), torch.bmm(Q_uu, K))  # Shape: (batch_size, state_dim, state_dim)

                # Store gains
                K_traj[:, t, :, :] = K
                k_traj[:, t, :] = k

            # Forward pass with line search
            x_new_traj = torch.zeros_like(x_traj)
            x_new_traj[:, 0, :] = x0.to(device)
            u_new_traj = torch.zeros_like(u_traj)

            for t in range(T):
                dx = x_new_traj[:, t, :] - x_ref_traj[:, t, :]  # Shape: (batch_size, Nx)
                du = u_traj[:, t, :] - u_ref_traj[:, t, :]  # Shape: (batch_size, Nu)
                # Compute control update
                delta_u = torch.bmm(K_traj[:, t, :, :], dx.unsqueeze(2)).squeeze(2) + k_traj[:, t, :]  # Shape: (batch_size, Nu)
                u_new = u_traj[:, t, :] + alpha * delta_u  # Shape: (batch_size, Nu)
                u_new_traj[:, t, :] = u_new

                # Simulate next state
                x_new_traj[:, t+1, :] = self.step(x_new_traj[:, t, :], u_new).detach()
            # Compute new cost
            new_cost = torch.zeros(batch_size, device=device)
            for t in range(T):
                dx = x_new_traj[:, t, :] - x_ref_traj[:, t, :]  # Shape: (batch_size, Nx)
                du = u_new_traj[:, t, :] - u_ref_traj[:, t, :]  # Shape: (batch_size, Nu)
                new_cost += torch.sum(dx @ self.Q * dx, dim=1) + torch.sum(du @ self.R * du, dim=1)
            dx_final = x_new_traj[:, -1, :] - x_ref_traj[:, -1, :]
            new_cost += torch.sum(dx_final @ self.Qf * dx_final, dim=1)
            # Check convergence
            cost_diff = torch.abs(new_cost - prev_cost)
            if torch.mean(cost_diff) < tol:
                print(f"Converged at iteration {iteration} at cost {new_cost}")
                return u_new_traj, x_new_traj

            # Update trajectories
            u_traj = u_new_traj.clone()
            x_traj = x_new_traj.clone()
            prev_cost = new_cost
        return u_traj, x_traj

    def plot_solver_performance(self, batch_size, generate_training_data_func, T, dynamics, controller, max_iters=100, tol=1e-2, alpha=0.1):
        """
        Validates the ILQR solver by computing the tracking error and plotting the reference and optimized trajectories in XY plane.

        Parameters:
            - batch_size (int): Size of the validation batch.
            - generate_training_data_func (callable): Function to generate validation data.
            - T (int): Time horizon.
            - dynamics: Dynamics model.
            - controller: Controller (ILQRHelper instance).
        """
        # Generate validation data
        r_traj_val, delta_r = generate_training_data_func(batch_size, T, dynamics, controller)
        
        # Initial state: assuming zero initial state
        x0 = r_traj_val[:, 0, :]

        # Reference control trajectory: assuming zero control by default
        u_ref_traj = torch.zeros(batch_size, T, dynamics.Nu, device=self.device)  # Shape: (batch_size, T, control_dim)
        u0 = torch.zeros(batch_size, T, dynamics.Nu, device=self.device)  # Initial control sequence

        # Solve iLQR to track the reference trajectory
        u_exac_unadj, x_exac_unadj = self.solve(x0, u0, r_traj_val, u_ref_traj, max_iters=max_iters, tol=tol, alpha=alpha) 

        # Compute tracking error
        tracking_error_unadj = torch.mean(delta_r)
        print(f'Tracking Error before adjustment: {tracking_error_unadj.item():.4f}')

        # Plot trajectories for a subset of the batch
        num_plots = min(batch_size, 5)  # Plot first 5 samples to avoid clutter

        for i in range(num_plots):
            plt.figure(figsize=(8, 6))
            # Extract X and Y positions from reference and optimized trajectories
            # Assuming first two dimensions are X and Y
            ref_x = r_traj_val[i, :, 0].cpu().numpy()
            ref_y = r_traj_val[i, :, 1].cpu().numpy()
            opt_x = x_exac_unadj[i, :, 0].cpu().numpy()
            opt_y = x_exac_unadj[i, :, 1].cpu().numpy()

            # Plot reference trajectory
            plt.plot(ref_x, ref_y, label='Reference Trajectory', linestyle='--', color='blue')

            # Plot optimized trajectory
            plt.plot(opt_x, opt_y, label='Optimized Trajectory', linestyle='-', color='orange')

            # Plot start and end points
            plt.scatter(ref_x[0], ref_y[0], color='green', marker='o', label='Start Point')
            plt.scatter(ref_x[-1], ref_y[-1], color='red', marker='x', label='End Point')

            plt.title(f'Trajectory Comparison for Sample {i+1}')
            plt.xlabel('X Position')
            plt.ylabel('Y Position')
            plt.legend()
            plt.grid(True)
            plt.axis('equal')  # Ensure equal scaling for X and Y axes
            plt.show()

