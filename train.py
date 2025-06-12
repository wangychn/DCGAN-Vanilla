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

from generator import Generator
from discriminator import Discriminator

from plot import training_process_plot

# <===========- Meta Values -============>

# Data loading
dataroot = "data"
model_folder_path = "./model"
output_path = "./outputs"
file_name = "model.pth"
plot_path = "plots"
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


def save_img_list(img_list):
    os.makedirs(output_path, exist_ok=True)
    for idx, img in enumerate(img_list):
        torchvision.utils.save_image(img, os.path.join(output_path, f"img_{idx:04d}.png"))


def train(discriminator, generator):

    trainset = dset.ImageFolder(root=dataroot, 
                                transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.ToTensor(), 
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ]))

    dataloader = torch.utils.data.DataLoader(trainset, 
                                            batch_size=batch_size, 
                                            shuffle=True,
                                            num_workers=num_workers
                                            ) 

    load_data(dataloader, device)

    criterion = nn.BCELoss()

    # Batch of latent random noise to visualize training progress
    fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))

    # <=============- Training loop -==============>

    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")

    for epoch in range(num_epochs):
        for i, data in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            
            # <================- Training Discriminator -=================>
            # Update D network: maximize log(D(x)) + log(1 - D(G(z)))

            # <=========- Train with all-real batch
            discriminator.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)

            # 1D tensor of batch size with target label, which is all correct (1) 
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            
            output_real = discriminator(real_cpu).view(-1)

            errorD_real = criterion(output_real, label)

            errorD_real.backward()
            disc_real_mean = output_real.mean().item()

            # <=========- Train with all-fake batch

            noise = torch.randn(b_size, latent_dim, 1, 1, device=device)

            fake = generator(noise)
            label.fill_(fake_label)

            # Classify the fake generations
            output_fake = discriminator(fake.detach()).view(-1)

            errorD_fake = criterion(output_fake, label)

            # NOTE: This is acculumated with previous gradients
            errorD_fake.backward()
            disc_fake_mean_pre = output_fake.mean().item()
            errorD = errorD_fake + errorD_real # for stats

            # Update D
            optimizerD.step()

            # <================- Training Generator -=================>
            # Update G network: maximize log(D(G(z)))

            generator.zero_grad()

            label.fill_(real_label)

            # Get new classifications for fake (the one we already made)
            output = discriminator(fake).view(-1)

            errorG = criterion(output, label)

            errorG.backward()
            disc_fake_mean_post = output.mean().item()

            optimizerG.step()

            # <=================- Training Stats -==================>
            
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(dataloader),
                        errorD.item(), errorG.item(), disc_real_mean, 
                        disc_fake_mean_pre, disc_fake_mean_post))

            G_losses.append(errorG.item())
            D_losses.append(errorD.item())

            # PLOT THE TRAINING PROCESS
            training_process_plot(G_losses, D_losses, plot_path=plot_path)

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = generator(fixed_noise).detach().cpu()
                img_list.append(torchvision.utils.make_grid(fake, padding=2, normalize=True))

            iters += 1

    training_process_plot(G_losses, D_losses, plot_path=plot_path, plot_name="training_plot.png")
    return img_list, G_losses, D_losses



def main():

    manualSeed = 445
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.use_deterministic_algorithms(True)

    os.makedirs(dataroot, exist_ok=True)
    os.makedirs(model_folder_path, exist_ok=True)


    file_path = os.path.join(model_folder_path, file_name)

    img_list = []
    G_losses = []
    D_losses = []

    netG = Generator(pic_dim=pic_dim, feat_dim=feat_dim, latent_dim=latent_dim)
    netD = Discriminator(pic_dim=pic_dim, feat_dim=feat_dim)

    img_list, G_losses, D_losses = train(netD, netG)

    # Save the model
    torch.save({
        'epoch': num_epochs,
        'modelG_state_dict': netG.state_dict(),
        'modelD_state_dict': netD.state_dict(),
        'optimizerG_state_dict': optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999)).state_dict(),
        'optimizerD_state_dict': optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999)).state_dict(),
        'lossG': G_losses,
        'lossD': D_losses
    }, file_path)

    save_img_list(img_list)





if __name__ == "__main__":
    main()