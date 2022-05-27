import torch
from torch import nn
torch.manual_seed(0)


def get_generator_block(input_dim, output_dim):
    """
    create a single sequential block of the generator consisting of 
    * Linear Layer
    * BatchNormalization Layer
    * ReLU activation Layer
    arguments: 
      input_dim: int
        spatial input dimension of the image
      output_dim: int
        spatial output dimension of the image
    returns: 
      nn.Sequential object constituting the generator block
    """
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.ReLU(inplace=True)
    )


def get_noise(n_samples, z_dim, device='cpu'):
    """
    generate a noise tensor as an input for the generator

    arguments: 
        n_samples: int
            number of sample images
        z_dim: int
            dimension of the input noise vector
        device: string -> cpu/cuda
            device to train the model on
    return:
        z_tensor: tf_tensor
            the generated noise tensor with the required dimension
    """
    return torch.randn((n_samples, z_dim), device=device)


class Generator(nn.Module):
    def __init__(self, z_dim=10, im_dim=784, hidden_dim=128):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            get_generator_block(z_dim, hidden_dim),
            get_generator_block(hidden_dim, hidden_dim * 2),
            get_generator_block(hidden_dim * 2, hidden_dim * 4),
            get_generator_block(hidden_dim * 4, hidden_dim * 8),

            nn.Linear(hidden_dim * 8, im_dim),
            nn.Sigmoid()
        )

    def forward(self, noise):
        return self.gen(noise)

    def get_gen(self):
        return self.gen


def get_discriminator_block(input_dim, output_dim):
    """
    create a single sequential block of the discriminator consisting of 
    * Linear Layer
    * LeakyReLU Activation Layer
    arguments: 
      input_dim: int
        spatial input dimension of the image
      output_dim: int
        spatial output dimension of the image
    returns: 
      nn.Sequential object constituting the discriminator block
    """
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.LeakyReLU(negative_slope=0.2, inplace=True)
    )


class Discriminator(nn.Module):
    def __init__(self, im_dim=784, hidden_dim=128):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            get_discriminator_block(im_dim, hidden_dim * 4),
            get_discriminator_block(hidden_dim * 4, hidden_dim * 2),
            get_discriminator_block(hidden_dim * 2, hidden_dim),

            nn.Linear(hidden_dim, 1)
        )

    def forward(self, image):
        return self.disc(image)

    def get_disc(self):
        return self.disc
