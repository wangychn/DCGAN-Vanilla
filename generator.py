import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

class Generator(nn.Module):
    def __init__(self, pic_dim, feat_dim, latent_dim):
        super(Generator, self).__init__()

        # Dimension of Generator:
        # input is latent ``(latent_dim) -> 100 x 1 x 1``
        # state size. ``(feat_dim * 8) x 4 x 4``
        # state size. ``(feat_dim * 8) x 8 x 8``
        # state size. ``(feat_dim * 4) x 16 x 16``
        # state size. ``(feat_dim * 2) x 32 x 32``
        # output is image. ``(pic_dim) -> 3 x 64 x 64``

        self.main = nn.Sequential(
            # Input is Z, the latent dim
            nn.ConvTranspose2d(latent_dim, feat_dim * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feat_dim * 8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(feat_dim * 8, feat_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_dim * 4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(feat_dim * 4, feat_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_dim * 4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(feat_dim * 2, feat_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(feat_dim, pic_dim, 4, 2, 1, bias=False),
            nn.Tanh(),
            # Output is c, 3 for channel of images
        )

        self.apply(self.weights_init)

    def forward(self, input):
        return self.main(input)
    
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)


if __name__ == "__main__":
    # ======= Example usage for Generator
    pic_dim = 3
    feat_dim = 64
    latent_dim = 100

    generator = Generator(pic_dim, feat_dim, latent_dim)

    print(generator)

    