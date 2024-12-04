# nu_mlp.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard's SummaryWriter
import matplotlib.pyplot as plt

# Define the Neural Network Model
class NuMLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64):
        super(NuMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

class NuMLPHelper:
    """
    Helper class for training and validating the NuMLP model.
    """
    def __init__(self, model, criterion, optimizer, dynamics, controller, horizon, device, log_dir='runs/nu_mlp'):
        """
        Initializes the NuMLPHelper.

        Parameters:
            model (NuMLP): The neural network model to train.
            criterion (nn.Module): Loss function.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            dynamics (LinearDynamics): Dynamics of the system.
            controller (LQTrackingController): Tracking controller.
            horizon (int): Planning horizon.
            device (torch.device): Device to run computations on.
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.dynamics = dynamics
        self.controller = controller
        self.horizon = horizon
        self.device = device

        # Initialize TensorBoard SummaryWriter
        self.writer = SummaryWriter(log_dir=log_dir)

    def train_model(self, num_epochs, batch_size, generate_training_data_func, checkpoint_interval=10):
        """
        Trains the NuMLP model.

        Parameters:
            num_epochs (int): Number of training epochs.
            batch_size (int): Size of each training batch.
            generate_training_data_func (callable): Function to generate training data.
            checkpoint_interval (int): Interval (in epochs) to save model checkpoints.
        """
        self.model.train()
        for epoch in range(1, num_epochs + 1):
            # Generate training data
            x_exac, delta_r = generate_training_data_func(batch_size, self.horizon, self.dynamics, self.controller)
            
            # Flatten the reference trajectories and delta_r for the neural network
            x_exac_flat = x_exac.reshape(batch_size, -1).to(self.device)  # Shape: (batch_size, Nx * horizon)
            delta_r_flat = delta_r.reshape(batch_size, -1).to(self.device)  # Shape: (batch_size, Nx * horizon)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(x_exac_flat)
            loss = self.criterion(outputs, delta_r_flat)
            
            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

            # Logging to TensorBoard
            self.writer.add_scalar('Loss/train', loss.item(), epoch)
            if epoch % 50 == 0 or epoch == 1:
                print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}')
            # Logging
            if epoch % checkpoint_interval == 0 or epoch == 1:
                # Save model checkpoint
                checkpoint_path = f'NuMLP_epoch{epoch}.pth'
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f'Model checkpoint saved at {checkpoint_path}')
                
                # Optionally, log model graph once
                if epoch == 1:
                    self.writer.add_graph(self.model, x_exac_flat)

    def load_model(self, checkpoint_path):
        """
        Loads the model weights from a checkpoint file.

        Parameters:
            checkpoint_path (str): Path to the checkpoint file.
        """
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.to(self.device)
        print(f'Model loaded from {checkpoint_path}')

    def validate_model(self, batch_size, generate_training_data_func, max_iters, tol, alpha, n_plt = 5):
        """
        Validates the NuMLP model by computing the tracking error.

        Parameters:
            batch_size (int): Size of the validation batch.
            generate_training_data_func (callable): Function to generate validation data.
        """
        self.model.eval()
        x_exac_unadj, delta_r = generate_training_data_func(batch_size, self.horizon, self.dynamics, self.controller)
        r_traj = x_exac_unadj + delta_r
        r_traj_flat = r_traj.reshape(batch_size, -1)
        delta_r_pred_flat = self.model(r_traj_flat)
        delta_r_pred = delta_r_pred_flat.reshape(batch_size, self.horizon + 1, self.dynamics.Nx)
        # Adjust reference trajectory
        adjusted_r_traj = r_traj + delta_r_pred  # shape (batch_size, T + 1, Nx)
        # Create x0_refs with adjusted reference trajectory
        x0 = r_traj[:, 0, :]  # Initial state is zero

        # Reference of control output is setting to be zero defaultly
        u_ref_traj = torch.zeros(batch_size, self.horizon, self.dynamics.Nu, device=self.device)
        u0 = torch.zeros(batch_size, self.horizon, self.dynamics.Nu, device=self.device)
            
        # Use controller to get control inputs with dual variable adjust
        u_exac, x_exac = self.controller.solve(x0, u0, adjusted_r_traj, u_ref_traj, max_iters=max_iters, tol=tol, alpha=alpha)

        # Loss
        Loss_ = torch.mean((x_exac_unadj - adjusted_r_traj) ** 2)
        print(f'Loss(unadjusted_xtraj - adjusted_rtraj): {Loss_.item():.4f}')

        # Compute tracking error before adjusting
        tracking_error_unadj = torch.mean((x_exac_unadj - r_traj) ** 2)
        print(f'Tracking Error before adjustment: {tracking_error_unadj.item():.4f}')

        # Compute tracking error between adjusted trajectory and original reference
        tracking_error = torch.mean((x_exac - r_traj) ** 2)
        print(f'Tracking Error after adjustment: {tracking_error.item():.4f}')
        # Plotting the trajectories for the first 5 samples
        num_samples_to_plot = min(n_plt, batch_size)  # Ensure we don't exceed the batch size

        time_steps = torch.arange(self.horizon + 1)

        for sample_idx in range(num_samples_to_plot):
            # Extract positions for plotting
            x_exac_pos = x_exac[sample_idx, :, :2].cpu().detach().numpy()
            x_exac_unadj_pos = x_exac_unadj[sample_idx, :, :2].cpu().detach().numpy()
            r_traj_pos = r_traj[sample_idx, :, :2].cpu().detach().numpy()
            adjusted_r_traj_pos = adjusted_r_traj[sample_idx, :, :2].cpu().detach().numpy()

            # Plot trajectories in 2D space
            plt.figure(figsize=(30,6))
            plt.plot(x_exac_pos[:, 0], x_exac_pos[:, 1], '-o', label='x_exac (Adjusted)')
            plt.plot(x_exac_unadj_pos[:, 0], x_exac_unadj_pos[:, 1], '-o', label='x_exac_unadj (Unadjusted)')
            plt.plot(r_traj_pos[:, 0], r_traj_pos[:, 1], '-o', label='r_traj (Reference)')
            plt.plot(adjusted_r_traj_pos[:, 0], adjusted_r_traj_pos[:, 1], '-o', label='adjusted_r_traj (Adjusted Reference)')
            plt.title(f'Sample {sample_idx + 1}: Trajectories in 2D Space')
            plt.xlabel('X Position')
            plt.ylabel('Y Position')
            plt.legend()
            plt.grid(True)
            plt.show()

    def close_writer(self):
        self.writer.close()