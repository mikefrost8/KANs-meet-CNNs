import torch
import torch.nn as nn

# ReLU
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

# Sigmoid
class LearnableActivationSigmoid(nn.Module):
    def __init__(self, in_features, hidden_features):
        super(LearnableActivationSigmoid, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        return x
        
