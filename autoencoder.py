import torch
from torch import nn

from ste import BinarizeSTE, ClampSTE, RoundSTE

class Autoencoder(nn.Module):
    def __init__(self, STE, apply_sigmoid=True):
        super(Autoencoder, self).__init__()
        self.STE = STE
        self.apply_sigmoid = apply_sigmoid

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # B,  32, 16, 16
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # B,  64, 8, 8
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),  # B, 128, 4, 4
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )
        
        if apply_sigmoid:
            self.encoder.add_module('Sigmoid', nn.Sigmoid())

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),  # B,  64, 8, 8
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # B,  32, 16, 16
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  # B,  3, 32, 32
            nn.Sigmoid(),
        )

    def forward(self, x, encode=False, decode=False):
        if encode:
            x = self.encoder(x)
            x = self.STE.apply(x)
        elif decode:
            x = self.decoder(x)
        else:
            x = self.encoder(x)
            x = self.STE.apply(x)
            x = self.decoder(x)
        return x
