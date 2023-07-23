from autoencoder import Autoencoder
from ste import BinarizeSTE, ClampSTE, RoundSTE

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt

def load_cifar10(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    trainset = datasets.CIFAR10('data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testset = datasets.CIFAR10('data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)
    return trainloader, testloader

def train(model, trainloader, testloader, epochs, lr, device):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    test_losses = []
    for epoch in range(epochs):
        train_loss = 0
        for images, _ in trainloader:
            images = images.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(trainloader)
        train_losses.append(train_loss)
        test_loss = 0
        for images, _ in testloader:
            images = images.to(device)
            outputs = model(images)
            loss = criterion(outputs, images)
            test_loss += loss.item()
        test_loss /= len(testloader)
        test_losses.append(test_loss)
        print(f'Epoch {epoch + 1}/{epochs} - train loss: {train_loss:.4f} - test loss: {test_loss:.4f}')
    return train_losses, test_losses

def plot_losses(train_losses, test_losses):
    '''
    Plot the training and test losses.
    '''
    plt.plot(train_losses, label='train loss')
    plt.plot(test_losses, label='test loss')
    plt.legend()
    plt.show()

def plot_images(model, testloader, device):
    images, _ = next(iter(testloader))
    images = images.to(device)
    outputs = model(images)
    images = images.cpu().numpy()
    outputs = outputs.cpu().detach().numpy()
    for i in range(10):
        plt.subplot(2, 10, i + 1)
        plt.imshow(np.transpose(images[i], (1, 2, 0)))
        plt.subplot(2, 10, i + 11)
        plt.imshow(np.transpose(outputs[i], (1, 2, 0)))
    plt.show()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device "{device}"')
    trainloader, testloader = load_cifar10(128)
    model = Autoencoder(BinarizeSTE, apply_sigmoid=True).to(device)
    train_losses, test_losses = train(model, trainloader, testloader, epochs=10, lr=1e-3, device=device)
    plot_losses(train_losses, test_losses)
    plot_images(model, testloader, device)

if __name__ == '__main__':
    main()