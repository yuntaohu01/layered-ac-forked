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
                 nominal_obs_dim=4, num_layers=2, nhead=8, dim_feedforward=256, 
                 dropout=0.1, max_seq_length=16):
        super().__init__()
        self.nominal_controller = nominal_controller

        self.nominal_obs_dim = nominal_obs_dim  # e.g., 4 (x, y, speed, theta)
        self.observation_dim = observation_dim  # total_dim = seq_length * nominal_obs_dim

        # Compute sequence length based on observation_dim and nominal_obs_dim
        if observation_dim % nominal_obs_dim != 0:
            raise ValueError(f"observation_dim ({observation_dim}) is not divisible by nominal_obs_dim ({nominal_obs_dim}).")
        self.seq_length = observation_dim // nominal_obs_dim

        # Positional Encoding (required for Transformer)
        self.positional_encoding = PositionalEncoding(d_model=self.nominal_obs_dim, max_len=max_seq_length)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.nominal_obs_dim,
                                                   nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   activation='relu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layers
        self.fc_mu = nn.Linear(self.nominal_obs_dim, action_dim)

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
        # Check if observation_dim matches
        if x.shape[1] != self.observation_dim:
            raise ValueError(f"Expected input dimension {self.observation_dim}, but got {x.shape[1]}.")

        # Reshape x to (batch_size, seq_length, nominal_obs_dim)
        x = x.view(x.size(0), self.seq_length, self.nominal_obs_dim)  # Shape: (batch, seq, nominal_obs_dim)

        # Apply positional encoding
        x = self.positional_encoding(x)  # Shape: (batch, seq, nominal_obs_dim)

        # Transformer expects input shape: (seq, batch, feature)
        x = x.permute(1, 0, 2)  # Shape: (seq, batch, feature)

        # Pass through Transformer Encoder
        x = self.transformer_encoder(x)  # Shape: (seq, batch, feature)

        # Take the output from the last time step
        x = x[-1, :, :]  # Shape: (batch, feature)

        # Compute action
        x = torch.tanh(self.fc_mu(x))  # Shape: (batch, action_dim)

        # Rescale actions
        return x * self.action_scale + self.action_bias  # Shape: (batch, action_dim)

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
        if seq_length > self.pe.size(1):
            raise ValueError(f"Sequence length {seq_length} exceeds maximum length {self.pe.size(1)}.")
        x = x + self.pe[:, :seq_length, :]
        return x
