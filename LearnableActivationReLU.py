import torch
import torch.nn as nn

class LearnableActivationReLU(nn.Module):
    def __init__(self, num_channels):
        super(LearnableActivationReLU, self).__init__()
        # Initialize weights and biases for each channel, keeping spatial dimensions in mind
        self.weight = nn.Parameter(torch.randn(1, num_channels, 1, 1))  # Shape adapted for broadcasting
        self.bias = nn.Parameter(torch.randn(1, num_channels, 1, 1))
        self.relu = nn.ReLU()

    def forward(self, x):
        # Apply the parameters across all spatial dimensions using broadcasting
        return self.relu(x * self.weight + self.bias)
