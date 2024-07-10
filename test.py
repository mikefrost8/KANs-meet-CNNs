import yaml

import torch
import torch.nn as nn
import torch.optim as optim

import torchmetrics

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

import importlib

import os
import time

import pandas as pd

def train(config):
    # Load and transform test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])
    
    num_workers = os.cpu_count() - 1
    
    data_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=100, shuffle=False, num_workers=num_workers)
    
    # Initialize device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model_dyn = importlib.import_module(config['model']['module_path'])
    model_class = getattr(model_dyn, config['model']['type'])
    model = model_class()
    
    weights_path = os.path.join('experiments', config['experiment']['name'], 'model.pth')
    weights = torch.load('experiments\experiment_1\model.pth')
    model.load_state_dict(weights)
    model.eval()
    model.to(device)
    
    # Compute accuracy on test set of CIFAR-10
    metric = torchmetrics.Accuracy(num_classes=10, average='macro', task='multiclass')
    
    start_time = time.time()
    
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            metric.update(predictions, labels)

    accuracy = metric.compute()
    end_time = time.time()
    test_time = end_time - start_time
    print(f'Accuracy: {accuracy*100:.2f}%')
    print(f'Time: {test_time:.2f}')
    
    # Store results
    results_df = pd.DataFrame({'Accuracy': [accuracy], 'Time': [test_time]})
    path = os.path.join('experiments', config['experiment']['name'], 'test_results')
    results_df.to_csv(path, index=False)

def main():
    # Load config file
    with open('config_test.yaml', 'r') as file:
        config = yaml.safe_load(file)
    train(config)

if __name__ == '__main__':
    main()