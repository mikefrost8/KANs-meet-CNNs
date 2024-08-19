import torch.nn as nn

class KANCNN(nn.Module):
    def __init__(self, num_classes=10, activation_funcs=None):
        super(KANCNN, self).__init__()
        
        if activation_funcs is None or len(activation_funcs) != 7:
            raise ValueError("activation_funcs must be a list with 7 elements.")
        
        # Initialize convolutional layers with placeholders for activation functions
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.act1 = activation_funcs[0]

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.act2 = activation_funcs[1]

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.act3 = activation_funcs[2]

        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.act4 = activation_funcs[3]

        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.act5 = activation_funcs[4]

        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Initialize classifier layers
        self.fc1 = nn.Linear(256 * 4 * 4, 4096)
        self.act6 = activation_funcs[5]

        self.fc2 = nn.Linear(4096, 4096)
        self.act7 = activation_funcs[6]

        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.act3(x)
        
        x = self.conv4(x)
        x = self.act4(x)
        
        x = self.conv5(x)
        x = self.act5(x)
        x = self.pool3(x)
        
        x = x.view(x.size(0), -1)  # Flatten the tensor for the classifier
        print("Shape after flattening:", x.shape)


        x = self.fc1(x)
        x = self.act6(x)
        
        x = self.fc2(x)
        x = self.act7(x)
        
        x = self.fc3(x)
        
        return x
