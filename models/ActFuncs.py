import torch.nn as nn

# ReLU
class LearnableActivationReLU(nn.Module):
    def __init__(self, in_features, hidden_features):
        super(LearnableActivationReLU, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

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
        
