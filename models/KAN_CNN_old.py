import torch
import torch.nn as nn

class KANCNN(nn.Module):
    def __init__(self, num_classes=10, activation_class=None):
        super(KANCNN, self).__init__()

        if activation_class is None:
            activation_class = nn.ReLU

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size= 3, padding=1),
            activation_class(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size= 3, padding=1),
            activation_class(192, 384),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size= 3, padding=1),
            activation_class(384, 768),
            nn.Conv2d(384, 256, kernel_size= 3, padding=1),
            activation_class(256, 512),
            nn.Conv2d(256, 256, kernel_size= 3, padding=1),
            activation_class(256, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 4096),
            activation_class(4096, 8192),
            nn.Linear(4096, 4096),
            activation_class(4096, 8192),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 4 * 4)
        x = self.classifier(x)
        return x