import torch
from torch import nn

from ste import BinarizeSTE, ClampSTE, RoundSTE

class Autoencoder(nn.Module):
    def __init__(self, STE, apply_sigmoid=True):
        super(Autoencoder, self).__init__()
        self.STE = STE
        self.apply_sigmoid = apply_sigmoid
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(12, 24, kernel_size=5),
            nn.ReLU(),
        )
        if apply_sigmoid:
            self.encoder.add_module('Sigmoid', nn.Sigmoid())
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(24, 12, kernel_size=5),
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, kernel_size=5),
            nn.Sigmoid(),
        )

    def forward(self, x, encode=False, decode=False):
        if encode:
            x = self.encoder(x)
        elif decode:
            x = self.decoder(x)
        else:
            x = self.encoder(x)
            x = self.STE.apply(x)
            x = self.decoder(x)
        return x
