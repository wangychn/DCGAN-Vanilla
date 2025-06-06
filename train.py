import os
import random
import torch
import torch.nn as nn
from tqdm import tqdm

import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.utils.data
import torchvision.datasets as dset

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np


def load_data(dataloader, device):
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(torchvision.utils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()


# custom weights initialization called on ``netG`` and ``netD``
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

if __name__ == "__main__":

    manualSeed = 445
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.use_deterministic_algorithms(True)


    # <===========- Meta Values -============>

    dataroot = "data"
    image_size = 64
    batch_size = 128
    num_workers = 0

    # <======================================>
    os.makedirs(dataroot, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trainset = dset.ImageFolder(root=dataroot, 
                            transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.ToTensor(), 
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])
                            )

    dataloader = torch.utils.data.DataLoader(trainset, 
                                            batch_size=batch_size, 
                                            shuffle=True,
                                            num_workers=num_workers
                                            ) 

    load_data(dataloader, device)

    