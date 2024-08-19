import yaml

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

import importlib

import os
import time

import pandas as pd

from tqdm import tqdm
from models.ActFuncs import LearnableActivationReLU, LearnableActivationSigmoid

# Helper method
def get_activation_instance(activation_class_name, in_features, hidden_features):
    activation_classes = {
        'LearnableActivationReLU': LearnableActivationReLU,
        'LearnableActivationSigmoid': LearnableActivationSigmoid
    }
    activation_class = activation_classes[activation_class_name]
    return activation_class(in_features, hidden_features)

def train(config):

    model_type = config['model_type']
    model_info = config['models'][model_type]

    # Load and transform data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])

    num_workers = os.cpu_count() - 1

    data_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=config['training']['batch_size'], shuffle=True, num_workers=num_workers)

    # Initialize device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Dynamic activation class fetching based on configuration
    activation_instances = []
    if 'activation_params' in model_info:
        # This assumes activation_params is a list of dicts with in_features and hidden_features
        for act_param in model_info['activation_params']:
            activation_instance = get_activation_instance(
                model_info['activation_class'],
                act_param['in_features'],
                act_param['hidden_features']
            )
            activation_instances.append(activation_instance)

    # Initialize model
    model_dyn = importlib.import_module(model_info['module_path'])
    model_class = getattr(model_dyn, model_info['type'])

    # activation_funcs = [
    #     LearnableActivationReLU(64),
    #     LearnableActivationReLU(192),
    #     LearnableActivationReLU(384),
    #     LearnableActivationReLU(256),
    #     LearnableActivationReLU(256),
    #     LearnableActivationReLU(4096),
    #     LearnableActivationReLU(4096)
    # ]

    
    model = model_class(num_classes=10, activation_funcs=activation_instances)  # Assuming the model can take a list of activation instances
    #model = model_class(num_classes=10, activation_funcs=activation_funcs)

    model.to(device)


    # Set optimizer
    optimizer_class = getattr(optim, config['training']['optimizer']['type'])
    optimizer = optimizer_class(model.parameters(), **config['training']['optimizer']['parameters'])

    # Set criterion
    criterion_class = getattr(nn, config['training']['criterion']['type'])
    criterion = criterion_class()

    # Save epoch, loss and times
    epochs = []
    losses = []
    times_epochs = []
    total_training_time = 0.0


    # Training process
    for epoch in range(config['training']['epochs']):
        start_time_epoch = time.time()
        running_loss = 0.0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{config['training']['epochs']}", unit='batch') as pbar:
            for idx, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
                pbar.update(1)

        
        avg_loss = running_loss / len(train_loader)
        epoch_time = time.time() - start_time_epoch
        total_training_time += epoch_time
        print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')
        epochs.append(epoch+1)
        losses.append(avg_loss)
        times_epochs.append(epoch_time)

    # Save results and corresponding yaml file
    plt.plot(epochs, losses, marker='o')
    plt.ylim(0, 2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')

    name = config['results']['name']
    full_path = os.path.join('experiments', name)
    os.makedirs(full_path, exist_ok=True)

    filename='loss_per_epoch.pdf'
    file_path = os.path.join(full_path, filename)
    plt.savefig(file_path, format='pdf')

    # Save the yaml file
    yaml_path = os.path.join(full_path, 'config.yaml')
    with open(yaml_path, 'w') as file:
        yaml.safe_dump(config, file)

    # Save the model
    model_path = os.path.join(full_path, 'model.pth')
    torch.save(model.state_dict(), model_path)

    # Save epochs and losses to a CSV file
    results_df = pd.DataFrame({'Epoch': epochs, 'Loss': losses, 'Time(s)': times_epochs})
    results_csv_path = os.path.join(full_path, 'training_loss.csv')
    results_df.to_csv(results_csv_path, index=False)

    total_minutes = int(total_training_time // 60)
    total_seconds = int(total_training_time % 60)

    with open(results_csv_path, 'a') as f:
        f.write(f"\nTotal training time:, {total_minutes} minutes, {total_seconds} seconds\n")
    plt.show()