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

# Updated helper method
def get_activation_instance(activation_class_name, num_channels):
    activation_classes = {
        'LearnableActivationReLU': LearnableActivationReLU,
        'LearnableActivationSigmoid': LearnableActivationSigmoid  # Assuming similar implementation for sigmoid
    }
    activation_class = activation_classes[activation_class_name]
    return activation_class(num_channels)  # Initialize with channel number

def train(config):
    model_type = config['model_type']
    model_info = config['models'][model_type]

    # Load and transform data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])

    data_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=config['training']['batch_size'], shuffle=True, num_workers=os.cpu_count() - 1)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Initialize activation functions with proper channel numbers
    activation_instances = [
        get_activation_instance('LearnableActivationReLU', 64),
        get_activation_instance('LearnableActivationReLU', 192),
        get_activation_instance('LearnableActivationReLU', 384),
        get_activation_instance('LearnableActivationReLU', 256),
        get_activation_instance('LearnableActivationReLU', 256),
        get_activation_instance('LearnableActivationReLU', 4096),  # Assuming flattened or adapted
        get_activation_instance('LearnableActivationReLU', 4096)
    ]

    model_dyn = importlib.import_module(model_info['module_path'])
    model_class = getattr(model_dyn, model_info['type'])
    model = model_class(num_classes=10, activation_funcs=activation_instances)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    epochs = []
    losses = []
    times_epochs = []
    total_training_time = 0.0

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

    plt.plot(epochs, losses, marker='o')
    plt.ylim(0, 2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')
    plt.show()

    name = config['results']['name']
    full_path = os.path.join('experiments', name)
    os.makedirs(full_path, exist_ok=True)
    file_path = os.path.join(full_path, 'loss_per_epoch.pdf')
    plt.savefig(file_path, format='pdf')

    yaml_path = os.path.join(full_path, 'config.yaml')
    with open(yaml_path, 'w') as file:
        yaml.safe_dump(config, file)

    model_path = os.path.join(full_path, 'model.pth')
    torch.save(model.state_dict(), model_path)

    results_df = pd.DataFrame({'Epoch': epochs, 'Loss': losses, 'Time(s)': times_epochs})
    results_csv_path = os.path.join(full_path, 'training_loss.csv')
    results_df.to_csv(results_csv_path, index=False)

    total_minutes = int(total_training_time // 60)
    total_seconds = int(total_training_time % 60)
    with open(results_csv_path, 'a') as f:
        f.write(f"\nTotal training time:, {total_minutes} minutes, {total_seconds} seconds\n")
    plt.show()