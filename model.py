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
    def __init__(self, in_dim, feat_dim):
        super(Generator, self).__init__()

        # input is ``(nc) x 64 x 64``
        # state size. ``(ndf) x 32 x 32``
        # state size. ``(ndf*2) x 16 x 16``
        # state size. ``(ndf*4) x 8 x 8``
        # state size. ``(ndf*8) x 4 x 4``
        

        self.main = nn.Sequential(

            # Input is Z
            nn.ConvTranspose2d(in_dim, feat_dim * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feat_dim * 8),
            nn.ReLU(inplace=True),
        )
