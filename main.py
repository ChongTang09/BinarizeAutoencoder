from autoencoder import Autoencoder
from ste import BinarizeSTE, ClampSTE, RoundSTE

import os

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt

def denormalize(image):
    image = image * 0.5 + 0.5  # reverse normalization
    image = image.clip(0, 1)  # clamp to [0, 1]
    return image

class AEBinarizerTrainer:
    def __init__(self, model, lr=1e-3, device=None, model_path=None):
        self.model = model
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.model_path = model_path
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def load_cifar10(self, batch_size):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        trainset = datasets.CIFAR10('data', train=True, download=True, transform=transform)
        self.trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testset = datasets.CIFAR10('data', train=False, download=True, transform=transform)
        self.testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

    def train(self, epochs):
        self.model.train()
        train_losses = []
        test_losses = []
        best_loss = float('inf')  # Initialize best loss to infinity
        for epoch in range(epochs):
            train_loss = 0
            for images, _ in self.trainloader:
                images = images.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, images)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            train_loss /= len(self.trainloader)
            train_losses.append(train_loss)

            test_loss = 0
            self.model.eval()
            with torch.no_grad():
                for images, _ in self.testloader:
                    images = images.to(self.device)
                    outputs = self.model(images)
                    loss = self.criterion(outputs, images)
                    test_loss += loss.item()
            test_loss /= len(self.testloader)
            test_losses.append(test_loss)
            self.model.train()

            print(f'Epoch {epoch + 1}/{epochs} - train loss: {train_loss:.4f} - test loss: {test_loss:.4f}')

            # If this epoch's test loss is lower than the best loss seen so far, save the model
            if test_loss < best_loss:
                best_loss = test_loss
                self.save_model(self.model_path)

        return train_losses, test_losses

    def plot_losses(self, train_losses, test_losses):
        plt.figure()
        plt.plot(train_losses, label='train loss')
        plt.plot(test_losses, label='test loss')
        plt.legend()
        plt.show()
        plt.savefig('img/lose.png')

    def plot_images(self):
        images, _ = next(iter(self.testloader))
        images = images.to(self.device)
        self.model.eval()
        outputs = self.model(images)
        images = denormalize(images.cpu().numpy())
        outputs = denormalize(outputs.cpu().detach().numpy())
        plt.figure()
        for i in range(10):
            plt.subplot(2, 10, i + 1)
            plt.imshow(np.transpose(images[i], (1, 2, 0)))
            plt.subplot(2, 10, i + 11)
            plt.imshow(np.transpose(outputs[i], (1, 2, 0)))
        plt.show()
        plt.savefig('img/images.png')

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device "{device}"')

    trainer = AEBinarizerTrainer(Autoencoder(BinarizeSTE, apply_sigmoid=True), lr=1e-3, device=device, model_path='model.pth')
    trainer.load_cifar10(128)
    train_losses, test_losses = trainer.train(epochs=10)
    # trainer.plot_losses(train_losses, test_losses)
    # trainer.plot_images()
    # trainer.save_model('model.pth')  # Save the trained model


if __name__ == '__main__':
    main()