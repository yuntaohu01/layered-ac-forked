# nu_mlp.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard's SummaryWriter

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
            r_traj, delta_r = generate_training_data_func(batch_size, self.horizon, self.dynamics, self.controller)
            
            # Flatten the reference trajectories and delta_r for the neural network
            r_traj_flat = r_traj.reshape(batch_size, -1).to(self.device)  # Shape: (batch_size, Nx * horizon)
            delta_r_flat = delta_r.reshape(batch_size, -1).to(self.device)  # Shape: (batch_size, Nx * horizon)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(r_traj_flat)
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
                    self.writer.add_graph(self.model, r_traj_flat)

    def load_model(self, checkpoint_path):
        """
        Loads the model weights from a checkpoint file.

        Parameters:
            checkpoint_path (str): Path to the checkpoint file.
        """
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.to(self.device)
        print(f'Model loaded from {checkpoint_path}')
        
    def validate_model(self, batch_size, generate_training_data_func, max_iters, tol, alpha):
        """
        Validates the NuMLP model by computing the tracking error.

        Parameters:
            batch_size (int): Size of the validation batch.
            generate_training_data_func (callable): Function to generate validation data.
        """
        model.eval()
        with torch.no_grad():
            r_traj_val, _ = generate_training_data_func(batch_size, T, dynamics, controller)
            r_traj_flat = r_traj_val.reshape(batch_size, -1)
            delta_r_pred_flat = model(r_traj_flat)
            delta_r_pred = delta_r_pred_flat.reshape(batch_size, T, dynamics.Nx)
            # Adjust reference trajectory
            adjusted_r_traj = r_traj_val + delta_r_pred  # shape (batch_size, T + 1, Nx)
        # Create x0_refs with adjusted reference trajectory
        x0 = torch.zeros(batch_size, dynamics.Nx, device=device)  # Initial state is zero

        # Reference of control output is setting to be zero defaultly
        u_ref_traj = torch.zeros(batch_size, T, Nu, device=device)
        u0 = torch.zeros(batch_size, T, Nu, device=device)
            
        # Use controller to track unadjusted_reference
        u_exac_unadj, x_exac_unadj = controller.solve(x0, u0, r_traj, u_ref_traj, max_iters=max_iters, tol=tol, alpha=alpha) 
        # Use controller to get control inputs
        u_exac, x_exac = controller.controller.solve(x0, u0, r_traj, u_ref_traj, max_iters=max_iters, tol=tol, alpha=alpha)

        # Compute tracking error before adjusting
        tracking_error_unadj = torch.mean((x_exac_unadj - r_traj_val) ** 2)
        print(f'Tracking Error before adjustment: {tracking_error_unadj.item():.4f}')

        # Compute tracking error between adjusted trajectory and original reference
        tracking_error = torch.mean((x_exac - r_traj_val) ** 2)
        print(f'Tracking Error after adjustment: {tracking_error.item():.4f}')

        # Logging to TensorBoard
        if global_step is not None:
            self.writer.add_scalar('Error_after/validation', tracking_error, global_step)
            self.writer.add_scalar('Error_before/validation', tracking_error_unadj, global_step)
        else:
            self.writer.add_scalar('Error_after/validation', tracking_error)
            self.writer.add_scalar('Error_before/validation', tracking_error_unadj)

    def close_writer(self):
        self.writer.close()