import torch
import torch.nn.functional as F
import numpy as np

def gaussian_kernel(kernel_size, sigma):
    """
    Computes the 2D Gaussian kernel of the given size and standard deviation.
    """
    center = kernel_size // 2
    x = torch.arange(kernel_size).float() - center
    y = torch.arange(kernel_size).float() - center
    x, y = torch.meshgrid(x, y)
    kernel = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel.unsqueeze(0).unsqueeze(0)

def gaussian_blur(image, kernel_size, sigma, device):
    """
    Applies Gaussian blur to the input image.
    """
    kernel = gaussian_kernel(kernel_size, sigma).to(device)
    return F.conv2d(image, kernel, padding=kernel_size//2)

def generalized_gaussian_blur(image, kernel_size, beta):
    """
    Applies generalized Gaussian blur to the input image.
    """
    sigma = (kernel_size - 1) * 0.5
    kernel = torch.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    for i in range(kernel_size):
        for j in range(kernel_size):
            x = i - center
            y = j - center
            kernel[i, j] = torch.exp(-(x ** 2 + y ** 2) ** (beta / 2) / (2 * sigma ** beta))
    kernel /= kernel.sum()

    return F.conv2d(image, kernel.unsqueeze(0).unsqueeze(0), padding=center)

def plateau_shaped_blur(image, kernel_size, beta):
    """
    Applies plateau-shaped blur to the input image.
    """
    kernel = torch.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    for i in range(kernel_size):
        for j in range(kernel_size):
            x = i - center
            y = j - center
            d = (x ** 2 + y ** 2) ** 0.5
            if d <= beta:
                kernel[i, j] = 1
            elif beta < d and d < kernel_size:
                kernel[i, j] = (1 - (d - beta) / (kernel_size - beta)) ** 2
    kernel /= kernel.sum()
    return F.conv2d(image, kernel.unsqueeze(0).unsqueeze(0), padding=center)


def reconstruction_loss(output, target, device):
    """
    Computes the reconstruction loss between the output and target images.
    """
    kernel_choices = ["gaussian", "generalized_gaussian", "plateau_shaped"]
    kernel_probs = [0.7, 0.15, 0.15]
    kernel_size_choices = list(range(13, 22, 2))
    beta_range_gaussian = (0.5, 4)
    beta_range_plateau_shaped = (1, 2)
    sigma_range_gaussian = (0.1, 2)

    # Choose a random kernel and kernel size
    # kernel_choice = np.random.choice(kernel_choices, p=kernel_probs)
    kernel_choice = 'gaussian'
    kernel_size = np.random.choice(kernel_size_choices)

    # Choose beta or sigma for the kernel if applicable
    if kernel_choice == "generalized_gaussian":
        beta = torch.rand(1) * (beta_range_gaussian[1] - beta_range_gaussian[0]) + beta_range_gaussian[0] 
        sigma = (kernel_size - 1) * 0.5 * (2 ** (1 / beta) - 1)
    elif kernel_choice == "plateau_shaped":
        beta = torch.rand(1) * (beta_range_plateau_shaped[1] - beta_range_plateau_shaped[0]) + beta_range_plateau_shaped[0]
        sigma = (kernel_size - 1) * 0.5
    elif kernel_choice == "gaussian":
        sigma = torch.rand(1) * (sigma_range_gaussian[1] - sigma_range_gaussian[0]) + sigma_range_gaussian[0]
        beta = None
    else:
        raise NotImplementedError

    H, W = output.shape[-2:]
    output = output.reshape(-1, 1, H, W)
    target = target.reshape(-1, 1, H, W)
    # Apply the chosen kernel to both the output and target images
    if kernel_choice == "gaussian":
        output_blurred = gaussian_blur(output, kernel_size, sigma, device)
        target_blurred = gaussian_blur(target, kernel_size, sigma, device)
    elif kernel_choice == "generalized_gaussian":
        output_blurred = generalized_gaussian_blur(output, kernel_size, beta)
        target_blurred = generalized_gaussian_blur(target, kernel_size, beta)
    elif kernel_choice == "plateau_shaped":
        output_blurred = plateau_shaped_blur(output, kernel_size, beta)
        target_blurred = plateau_shaped_blur(target, kernel_size, beta)

    loss = F.mse_loss(output_blurred, target_blurred)
    return loss