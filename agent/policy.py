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
        
class TransformerUnicycleActor(nn.Module):
    """Learns a residual to the given nominal controller using a Transformer encoder."""
    def __init__(self, nominal_controller, observation_dim, action_dim, u_max, u_min, 
                 num_layers=2, nhead=8, dim_feedforward=256, dropout=0.1, embedding_dim=64):
        super().__init__()
        self.nominal_controller = nominal_controller

        # Validate observation_dim
        if observation_dim < 4:
            raise ValueError("observation_dim must be at least 4 to accommodate the initial state.")
        if (observation_dim - 4) % 2 != 0:
            raise ValueError("The remaining observation dimensions after the initial state must be divisible by 2.")

        self.num_ref_points = (observation_dim - 4) // 2
        self.actual_obs_dim = 2  # Each reference point has 2 dimensions (x, y)
        self.embedding_dim = embedding_dim

        # Projection layers
        self.project_initial = nn.Linear(4, embedding_dim)
        self.project_ref = nn.Linear(2, embedding_dim)

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model=self.embedding_dim, max_len=self.num_ref_points + 1)  # +1 for duplicated (x, y)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_dim,
                                                   nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   activation='relu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final layers
        self.fc_combined = nn.Linear(embedding_dim + embedding_dim, 256)  # Initial state + Transformer output
        self.fc_mu = nn.Linear(256, action_dim)

        # Action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((u_max - u_min) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((u_max + u_min) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, observation_dim)
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
        projected_initial = F.relu(self.project_initial(initial_state))  # Shape: (batch_size, embedding_dim)
        projected_ref = F.relu(self.project_ref(reference_points))       # Shape: (batch_size, num_ref_points +1, embedding_dim)

        # Apply positional encoding
        x_seq = self.positional_encoding(projected_ref)  # Shape: (batch_size, num_ref_points +1, embedding_dim)

        # Transformer expects input shape: (seq_length, batch_size, input_dim)
        x_seq = x_seq.permute(1, 0, 2)  # Shape: (num_ref_points +1, batch_size, embedding_dim)

        # Pass through Transformer Encoder
        x_seq = self.transformer_encoder(x_seq)  # Shape: (num_ref_points +1, batch_size, embedding_dim)

        # Take the output from the last time step
        x_seq = x_seq[-1, :, :]  # Shape: (batch_size, embedding_dim)

        # Combine projected initial state and Transformer output
        combined = torch.cat([projected_initial, x_seq], dim=1)  # Shape: (batch_size, 2 * embedding_dim)

        # Pass through final layers
        x = F.relu(self.fc_combined(combined))  # Shape: (batch_size, 256)
        x = torch.tanh(self.fc_mu(x))           # Shape: (batch_size, action_dim)

        # Rescale actions
        return x * self.action_scale + self.action_bias  # Shape: (batch_size, action_dim)

class PositionalEncoding(nn.Module):
    """Adds positional information to the input embeddings."""
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        # Create positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, seq_length, d_model)
        """
        seq_length = x.size(1)
        x = x + self.pe[:, :seq_length, :]
        return x
