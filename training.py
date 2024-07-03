import yaml

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

import importlib

import os

# Load config file
with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

# Load and transform data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
])

num_workers = os.cpu_count() - 1

data_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(data_train, batch_size=64, shuffle=True, num_workers=num_workers)

# Initialize model
model_dyn = importlib.import_module(config['model']['module_path'])
model_class = getattr(model_dyn, config['model']['type'])
model = model_class(input_size=config['model']['input_size'], num_classes=config['model']['num_classes'])

# Set optimizer
optimizer_class = getattr(optim, config['training']['optimizer']['type'])
optimizer = optimizer_class(model.parameters(), **config['training']['optimizer']['parameters'])

# Set criterion
criterion_class = getattr(nn, config['training']['criterion']['type'])
criterion = criterion_class()

# Save epoch and loss
epochs = []
losses = []

# Training process
for epoch in range(config['training']['epochs']):
    for idx, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    epochs.append(epoch)
    losses.append(loss)

# Save results and corresponding yaml file
plt.plot(epochs, loss)

name = config['results']['path']

full_path = os.path.join('experiments', name)
os.makedirs(full_path, exist_ok=True)

filename='loss_per_epoch.pdf'
file_path = os.path.join(full_path, filename)
plt.savefig(file_path, format='pdf')

yaml_path = os.path.join(full_path, 'config.yaml')

with open(yaml_path, 'w') as file:
    yaml.safe_dump(config, file)

plt.show()