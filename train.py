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


# <===========- Meta Values -============>

# Data loading
dataroot = "data"
image_size = 64
num_workers = 0

# Training
beta1 = 0.5
lr = 0.0002
num_epochs = 5
batch_size = 128


# Model
pic_dim = 3
feat_dim = 64
latent_dim = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# <======================================>


def load_data(dataloader, device):
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(torchvision.utils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()


def train(netD, netG):

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

    criterion = nn.BCELoss()

    # Batch of latent random noise to visualize training progress
    fixed_noise = torch.randn(64, pic_dim, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # <=============- Training loop -==============>

    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")

    for epoch in range(num_epochs):
        for i, data in enumerate(tqdm(dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}"):
            
            # <================- Training Discriminator -=================>
            # Update D network: maximize log(D(x)) + log(1 - D(G(z)))

            netD.zero_grad()


            # <================- Training Generator -=================>
            # Update G network: maximize log(D(G(z)))

            netG.zero_grad()



            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(torchvision.utils.make_grid(fake, padding=2, normalize=True))

            iters += 1



def main():

    manualSeed = 445
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.use_deterministic_algorithms(True)


    os.makedirs(dataroot, exist_ok=True)



if __name__ == "__main__":
    main()