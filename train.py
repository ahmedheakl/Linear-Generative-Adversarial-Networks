import torch
from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from model import *
from utils import *
torch.manual_seed(0)


# Hyperparameters Initialization
criterion = nn.BCEWithLogitsLoss()
n_epochs = 200
z_dim = 64
display_step = 500
batch_size = 128
lr = 0.001
device = 'cuda'

dataloader = DataLoader(MNIST('.', download=True, transform=transforms.ToTensor()),
                        batch_size=batch_size, shuffle=True)

gen = Generator(z_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
disc = Discriminator().to(device)
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)


def get_disc_loss(gen, disc, criterion, real, num_images, z_dim, device):
    """
    calculate the discriminator loss which is weighted average between the fake
    images loss and the real images loss(50%, 50%). 

    paramters
    ---------
      gen: tf_object
        generator network

      disc: tf_object
        discriminator network

      criterion: function
        the discriminator error function

      real: tf_tensor
        the real images from the dataset 

      num_images: int
        number of input images

      z_dim: int
        input dimension of the noise vector

      device: string -> cpu/cuda
        the device to train the model on 

    Returns
    --------------
      disc_loss: float 
        the discriminator calculated loss 
    """
    noise = get_noise(num_images, z_dim, device)
    fake_loss = criterion(disc(gen(noise).detach()),
                          torch.zeros(num_images, 1).to(device))
    real_loss = criterion(disc(real), torch.ones(num_images, 1).to(device))
    disc_loss = (fake_loss + real_loss) / 2
    return disc_loss


def get_gen_loss(gen, disc, criterion, num_images, z_dim, device):
    """
    calculate the generator loss, which is the absolute difference between 
    the expected labels and ones 

    paramters
    ---------
      gen: tf_object
        generator network

      disc: tf_object
        discriminator network

      criterion: function
        the discriminator error function

      real: tf_tensor
        the real images from the dataset 

      num_images: int
        number of input images

      z_dim: int
        input dimension of the noise vector

      device: string -> cpu/cuda
        the device to train the model on 

    Returns
    --------------
      disc_loss: float 
        the generator calculated loss 
    """
    noise = get_noise(num_images, z_dim, device)
    return criterion(disc(gen(noise)), torch.ones(num_images, 1).to(device))


def train():
    cur_step = 0
    mean_generator_loss = 0
    mean_discriminator_loss = 0
    gen_loss = False

    # the training loop
    for epoch in range(n_epochs):

        for real, _ in tqdm(dataloader):
            cur_batch_size = len(real)
            real = real.view(cur_batch_size, -1).to(device)

            # DISCRIMINATOR LOSS UPDATE
            disc_opt.zero_grad()
            disc_loss = get_disc_loss(
                gen, disc, criterion, real, cur_batch_size, z_dim, device)
            disc_loss.backward(retain_graph=True)
            disc_opt.step()

            # GENERATOR LOSS UPDATE
            gen_opt.zero_grad()
            gen_loss = get_gen_loss(gen, disc, criterion,
                                    cur_batch_size, z_dim, device)
            gen_loss.backward(retain_graph=True)
            gen_opt.step()

            mean_discriminator_loss += disc_loss.item() / display_step
            mean_generator_loss += gen_loss.item() / display_step

            if cur_step % display_step == 0 and cur_step > 0:
                print(
                    f"Step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
                fake_noise = get_noise(cur_batch_size, z_dim, device=device)
                fake = gen(fake_noise)
                show_images(fake)
                show_images(real)
                mean_generator_loss = 0
                mean_discriminator_loss = 0
            cur_step += 1

if __name__ == '__main__':
    print("Training ... ")
    train()
