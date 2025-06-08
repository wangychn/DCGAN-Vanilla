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

class Discriminator(nn.Module):
    def __init__(self, pic_dim, feat_dim):
        super(Discriminator, self).__init__()

            # input is image ``(pic_dim) -> 3 x 64 x 64``
            # state size. ``(feat_dim) x 32 x 32``
            # state size. ``(feat_dim * 2) x 16 x 16``
            # state size. ``(feat_dim * 4) x 8 x 8``
            # state size. ``(feat_dim * 8) x 4 x 4``
            # output is binary ``1 x 1 x 1``

        self.main = nn.Sequential(
            nn.Conv2d(pic_dim, feat_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(feat_dim, feat_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feat_dim * 2, feat_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feat_dim * 4, feat_dim * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feat_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feat_dim * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
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

    # ======= Example usage for Discriminator
    pic_dim = 3
    feat_dim = 64

    discriminator = Discriminator(pic_dim, feat_dim)

    print(discriminator)