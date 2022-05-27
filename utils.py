import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
torch.manual_seed(0)


def show_images(image_tensor, num_image=25, size=(1, 28, 28)):
    """
    show a blob of images on a grid 

    arguments:
      image_tensor: tf_tensor
        the input images in a tensor format
      num_images: int
        number of images to be visualized
      size: tuple (1,3)
        input dimension of each image  
    """
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_image], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


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
