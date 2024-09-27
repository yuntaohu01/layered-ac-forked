import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import unicycle_tracking_normalization

class SimpleActor(nn.Module):
    def __init__(self, observation_dim, action_dim, u_max, u_min):
        super().__init__()
        self.fc1 = nn.Linear(observation_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, action_dim)
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((u_max - u_min) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((u_max + u_min) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


class LinearActor(nn.Module):
    def __init__(self, observation_dim, action_dim):
        super().__init__()
        self.gain = nn.Linear(observation_dim, action_dim, bias=False)
        self.true_gain = None
        self.register_buffer(
            "action_scale", torch.tensor(1., dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor(0., dtype=torch.float32)
        )

    def forward(self, x):
        #true_u = x @ self.true_gain.T
        u = self.gain(x)
        return u


class UnicycleActor(nn.Module):
    """Learns a residual to the given nominal controller using an RNN (LSTM)."""
    def __init__(self, nominal_controller, observation_dim, action_dim, u_max, u_min,
                 hidden_size=256, num_layers=2, dropout=0.1, projection_dim=64):
        super().__init__()
        self.nominal_controller = nominal_controller

        # Validate observation_dim
        if observation_dim < 4:
            raise ValueError("observation_dim must be at least 4 to accommodate the initial state.")
        if (observation_dim - 4) % 2 != 0:
            raise ValueError("The remaining observation dimensions after the initial state must be divisible by 2.")

        self.num_ref_points = (observation_dim - 4) // 2
        self.actual_obs_dim = 2  # Each reference point has 2 dimensions (x, y)
        self.projection_dim = projection_dim

        # Projection layers
        self.project_initial = nn.Linear(4, projection_dim)
        self.project_ref = nn.Linear(2, projection_dim)

        # RNN (LSTM) layer
        self.lstm = nn.LSTM(input_size=self.projection_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout)

        # Final layers
        self.fc_combined = nn.Linear(projection_dim + hidden_size, 256)
        self.fc_mu = nn.Linear(256, action_dim)

        # Action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((u_max - u_min) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((u_max + u_min) / 2.0, dtype=torch.float32)
        )

    def forward(self, x, hidden=None):
        """
        x: Tensor of shape (batch_size, observation_dim)
        hidden: Tuple of (h_0, c_0) for LSTM initial hidden state (optional)
        """
        batch_size = x.size(0)

        # Separate initial state and reference points
        initial_state = x[:, :4]                     # Shape: (batch_size, 4)
        reference_points = x[:, 4:]                  # Shape: (batch_size, observation_dim - 4)

        # Reshape reference points to [batch_size, num_ref_points, 2]
        reference_points = reference_points.view(batch_size, self.num_ref_points, 2)  # Shape: (batch_size, num_ref_points, 2)

        # Extract and duplicate the initial (x, y)
        initial_xy = initial_state[:, :2].unsqueeze(1)  # Shape: (batch_size, 1, 2)
        reference_points = torch.cat([initial_xy, reference_points], dim=1)  # Shape: (batch_size, num_ref_points +1, 2)

        # Project initial state and reference points
        projected_initial = F.relu(self.project_initial(initial_state))  # Shape: (batch_size, projection_dim)
        projected_ref = F.relu(self.project_ref(reference_points))       # Shape: (batch_size, num_ref_points +1, projection_dim)

        # Pass through LSTM
        lstm_out, hidden = self.lstm(projected_ref, hidden)  # lstm_out: (batch_size, num_ref_points +1, hidden_size)

        # Take the output from the last time step
        lstm_last = lstm_out[:, -1, :]  # Shape: (batch_size, hidden_size)

        # Combine projected initial state and LSTM output
        combined = torch.cat([projected_initial, lstm_last], dim=1)  # Shape: (batch_size, projection_dim + hidden_size)

        # Pass through final layers
        x = F.relu(self.fc_combined(combined))  # Shape: (batch_size, 256)
        x = torch.tanh(self.fc_mu(x))           # Shape: (batch_size, action_dim)

        # Rescale actions
        return x * self.action_scale + self.action_bias  # Shape: (batch_size, action_dim)

