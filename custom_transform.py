"""
Most functions are obtained from https://kornia.readthedocs.io/en/latest/_modules/kornia/
The main difference is that this file contains the 
  - ElasticTransform class; and
  - a differentiable Gaussian filter.
"""

from typing import Tuple, Optional

import torch
import torch.nn as nn
import kornia
import math
import numpy as np
from torch.distributions import Uniform
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import scipy.ndimage.filters
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms.functional as F

class RandomHorizontalFlip(torch.nn.Module):
    def __init__(self, p =0.5) -> None:
        super().__init__()
        self.p = p
        self.requires_grad_(False)
        
    def forward(self, sample):
        import torchvision.transforms.functional as F
        
        if torch.rand(1) < self.p:
            result = {
            "image": F.hflip(sample["image"]),
            "mask" : F.hflip(sample["mask"])
        }
            return result
        return sample


class RandomVerticalFlip(torch.nn.Module):
    def __init__(self, p =0.5) -> None:
        super().__init__()
        self.p = p
        self.requires_grad_(False)

    def forward(self, sample):
        import torchvision.transforms.functional as F

        if torch.rand(1) < self.p:
            result = {
            "image": F.vflip(sample["image"]),
            "mask" : F.vflip(sample["mask"])
        }
            return result
        return sample


class ToTensor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.requires_grad_(False)

    def __call__(self, sample):
        import torchvision.transforms.functional as F

        result = {
            "image": F.to_tensor(sample["image"]),
            "mask" : F.to_tensor(sample["mask"])
        }
        return result


class ToPILImage:

    def __init__(self, mode=None):
        self.mode = mode
        

    def __call__(self, sample):
        import torchvision.transforms.functional as F
        
        result = {
            "image": F.to_pil_image(sample["image"], self.mode),
            "mask" : F.to_pil_image(sample["mask"], self.mode)
        }
        return result


class Resize(torch.nn.Module):
    def __init__(self, size, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=None):
        super().__init__()
        self.size = size
        self.max_size = max_size
        self.interpolation = interpolation
        self.antialias = antialias
        self.requires_grad_(False)

    def forward(self, sample):
        import torchvision.transforms.functional as F

        result = {
            "image": F.resize(sample["image"], self.size, self.interpolation, self.max_size,  self.antialias),
            "mask" : F.resize(sample["mask"], self.size, self.interpolation, self.max_size, self.antialias)
        }
        return result
 

class ElasticTransform(nn.Module):
    def __init__(self, alpha=1, sigma=12, random_seed=42):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.ones(
            2), requires_grad=True) * math.log(sigma)
        self.log_alpha = nn.Parameter(torch.ones(
            2), requires_grad=True) * math.log(alpha) # log e
        self.random_seed = random_seed
        self.requires_grad_(False)

    def forward(self, sample):
        # 원래 sigma, alpha로 만들어줌
        sigma = torch.exp(self.log_sigma)
        alpha = torch.exp(self.log_alpha)

        result = {
        "image": custom_transform_2d(sample["image"].unsqueeze(0), # image
                                            kernel_size=(3, 3),
                                            sigma=sigma,
                                            alpha=alpha,
                                            random_seed=self.random_seed).squeeze(0),
        "mask" : custom_transform_2d(sample["mask"].unsqueeze(0), # mask
                                            kernel_size=(3, 3),
                                            sigma=sigma,
                                            alpha=alpha,
                                            random_seed=self.random_seed).squeeze(0)
        }

        
        return result


def custom_transform_2d(tensor: torch.Tensor,
                         kernel_size: Tuple[int, int] = (3, 3),
                         sigma: Tuple[float, float] = (4., 4.),
                         alpha: Tuple[float, float] = (32., 32.),
                         random_seed= None) -> torch.Tensor:
    r"""Applies elastic transform of images as described in [Simard2003]_.
    Args:
        img (torch.Tensor): input image.
        kernel_size (Tuple[int, int]): the size of the Gaussian kernel. Default:(3,3).
        sigma (Tuple[float, float]): the standard deviation of the Gaussian in the y and x directions, respecitvely. 
                                     Larger sigma results in smaller pixel displacements. Default:(4,4).
        alpha (Tuple[float, float]):  the scaling factor that controls the intensity of the deformation
                                  in the y and x directions, respectively. Default:(32,32).
        random_seed (Optional): random seed for generating the displacement vector. Default:None.
    Returns:
        img (torch.Tensor): the elastically transformed input image.
    References:
        [Simard2003]: Simard, Steinkraus and Platt, "Best Practices for
                      Convolutional Neural Networks applied to Visual Document Analysis", in
                      Proc. of the International Conference on Document Analysis and
                      Recognition, 2003.
    """
    # https://runebook.dev/ko/docs/pytorch/generated/torch.generator
    generator = torch.Generator(device='cpu')
    if random_seed is not None:
        generator.manual_seed(random_seed)

    n, c, h, w = tensor.shape

    # Convolve over a random displacement matrix and scale them with 'alpha'
    d_rand = torch.rand(n, 2, h, w, generator=generator).to(
        device=tensor.device) * 2 - 1

    tensor_y = d_rand[:, 0] # shape [n, 1, h, w]
    tensor_x = d_rand[:, 1] # shape [n, 1, h, w]

    dy = apply_gaussian(tensor_y, sigma[0]) * alpha[0]
    dx = apply_gaussian(tensor_x, sigma[1]) * alpha[1]

    # stack and normalize displacement
    d_yx = torch.cat([dy, dx], dim=1).permute(0, 2, 3, 1)

    # Warp image based on displacement matrix
    # https://gaussian37.github.io/dl-pytorch-snippets/#fgrid_sample-%ED%95%A8%EC%88%98-%EC%82%AC%EC%9A%A9-%EC%98%88%EC%A0%9C-1
    grid = kornia.utils.create_meshgrid(h, w).to(device=tensor.device)
    warped = torch.nn.functional.grid_sample(
        tensor, (grid + d_yx).clamp(-1, 1), align_corners=True)
    return warped

# helpers
# ============


def apply_gaussian_numpy(tensor, sigma):
    t1 = scipy.ndimage.filters.gaussian_filter(
        tensor.numpy(), float(sigma), mode='constant')

    return t1


def apply_gaussian(tensor, sigma):
    kernel_size = int(2*(4.0*sigma+0.5))
    kernel_size = kernel_size if (kernel_size % 2 == 1) else kernel_size - 1
    k = get_gaussian_kernel2d((kernel_size, kernel_size), (sigma, sigma))
    t2 = kornia.filters.filter2d(
        tensor[:, None], kernel=k[None], border_type='constant')

    return t2


def gaussian(window_size, sigma):
    r"""
    Modified from Kornia to allow gradients to flow
    """
    x = torch.arange(window_size).float().to(
        device=sigma.device) - window_size // 2
    if window_size % 2 == 0:
        x = x + 0.5
    gauss = torch.exp((-x.pow(2.0) / (2 * sigma ** 2)))
    return gauss / gauss.sum()


def get_gaussian_kernel2d(
        kernel_size: Tuple[int, int],
        sigma: Tuple[float, float],
        force_even: bool = False) -> torch.Tensor:
    r"""
    See Kornia for Description
    """
    if not isinstance(kernel_size, tuple) or len(kernel_size) != 2:
        raise TypeError(
            "kernel_size must be a tuple of length two. Got {}".format(
                kernel_size
            )
        )
    if not isinstance(sigma, tuple) or len(sigma) != 2:
        raise TypeError(
            "sigma must be a tuple of length two. Got {}".format(sigma)
        )
    ksize_x, ksize_y = kernel_size
    sigma_x, sigma_y = sigma
    kernel_x: torch.Tensor = get_gaussian_kernel1d(
        ksize_x, sigma_x, force_even)
    kernel_y: torch.Tensor = get_gaussian_kernel1d(
        ksize_y, sigma_y, force_even)
    kernel_2d: torch.Tensor = torch.matmul(
        kernel_x.unsqueeze(-1), kernel_y.unsqueeze(-1).t()
    )
    return kernel_2d


def get_gaussian_kernel1d(kernel_size: int,
                          sigma: float,
                          force_even: bool = False) -> torch.Tensor:
    r"""
    See Kornia for Description
    """
    if (not isinstance(kernel_size, int) or (
            (kernel_size % 2 == 0) and not force_even) or (
            kernel_size <= 0)):
        raise TypeError(
            "kernel_size must be an odd positive integer. "
            "Got {}".format(kernel_size)
        )
    window_1d: torch.Tensor = gaussian(kernel_size, sigma)
    return window_1d