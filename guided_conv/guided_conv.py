from typing import Tuple, Optional, Union

import functools
import operator

from itertools import repeat

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


def _expand_kernel(
    kernel_size: Union[int, Tuple[int,...]],
    n: int = 2
) -> Tuple:

    """
    Expand an integer kernel size to tuple

    Arguments:

        kernel_size : int or tuple of ints that determines the kernel's spatial dimensions
        
        n : kernel dimensions, default to 2

    Returns:

        kernel_size : a tuple
    """

    if isinstance(kernel_size, int):
        kernel_size = tuple(repeat(kernel_size, n))
    
    return kernel_size


def _check_odd_kernel(
    kernel_size: Tuple[int,...]
) -> None:

    """
    Verifies that the kernel size is odd, raises ValueError

    Arguments:

        Kernel_size : tuple of integers
    """

    if not all([x % 2 != 0 for x in kernel_size]):
        raise ValueError(f'Expected odd kernel size, got {kernel_size}.')


def _get_padding(
    kernel_size: Tuple[int,...]
) -> Tuple:

    """
    Returns the padding such that the spatial dimensions 
    of the convolution output are identical to the input dimensions.

    Arguments:
        kernel_size : tuple of integers

    Return:
        tuple, padding per dimension
    """
    
    return tuple([x // 2 for x in kernel_size])


def _get_kernel_numel(
    kernel_size: Tuple[int,...]
) -> Tuple:

    """
    Returns the numer of kernel elements.

    Arguments:
        kernel_size : tuple of integers

    Return:
        number of kernel elements
    """

    return functools.reduce(operator.mul, kernel_size)


class _GuidedConvNd(nn.Module):

    """
    Base class for guided convolutional layers.
    """

    def __init__(
        self, 
        input_channels: int,
        output_channels: Optional[int] = None,
        kernel_size: Optional[Tuple[int,...]] = None
    ) -> None:

        """
        Arguments:
            input_channels : number of input channels, Ci

            output_channels : (optional) number of output channels, Co

            kernel_size : (optional) kernel size, tuple of integers, 
                          must be odd (not necessarily equal), e.g., (1,1), (3,3), (3,5), etc.
        """

        super().__init__()

        self._input_channels = input_channels
        self._output_channels = output_channels
        self._kernel_size = kernel_size

        self._kernel_numel = None

        if kernel_size is not None:
            _check_odd_kernel(kernel_size)
            self._kernel_numel = _get_kernel_numel(kernel_size)


class GuidedChannelWiseConv2d(_GuidedConvNd):

    """
    A guided channelwise 2D convolutional layer from
    "Learning Guided Convolutional Network for Depth Completion", Tang et al., 2019

    The convolution uses different weights for every site in every sample.
    """

    def __init__(
        self,
        input_channels: int,
        kernel_size: Union[int, Tuple[int,int]],
    ) -> None:

        """
        Arguments:
            input_channels : number of input channels, Ci

            kernel_size : kernel size, tuple of integers, 
                          must be odd (not necessarily equal), e.g., (1,1), (3,3), (3,5), etc.
        """

        # Expand kernel size
        kernel_size = _expand_kernel(kernel_size, 2)

        super().__init__(        
            input_channels=input_channels,
            output_channels=None,
            kernel_size=kernel_size
        )

        # Init params
        _input_padding = _get_padding(kernel_size)
        self._unfold_params = dict(kernel_size=kernel_size, padding=_input_padding)

    def forward(
        self,
        input: Tensor,
        weights: Tensor
    ) -> Tensor:

        """
        Arguments:
            input : input image, (B,Ci,H,W)

            weights : convolution weights, (B,Ci,K1xK2,H,W)

        Return:
            Convolved image, number of input channels and spatial dimensions are not changed (B,Ci,H,W)
        """

        # Extract params
        B, Ci, H, W = input.shape
        K = self._kernel_numel

        # Channelwise conovlution
        w = weights.reshape((B,Ci*K,-1))  # (B,Ci x K,HxW)
        x = F.unfold(input=input, **self._unfold_params)  # (B,Ci x K,HxW)
        x = x * w  # (B,Ci x K,HxW)
        x = x.reshape((B,Ci,K,-1)).sum(dim=2)  # (B,Ci x K,HxW) -> (B,Ci,K,HxW) -> (B,Ci,HxW)
        out = F.fold(input=x, output_size=(H,W), kernel_size=(1,1), padding=(0,0))  # (B,Ci,H,W)
        
        return out


class GuidedDepthWiseConv2d(_GuidedConvNd):

    """
    A guided depthwise (1x1) 2D convolutional layer from
    "Learning Guided Convolutional Network for Depth Completion", Tang et al., 2019

    The convolution uses different wieghts per sample in batch
    and are shared across all sites within that sample.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int
    ) -> None:
        
        """
        Arguments:
            input_channels : number of input channels, Ci

            output_channels : number of output channels, Co
        """

        super().__init__(        
            input_channels=input_channels,
            output_channels=output_channels,
            kernel_size=(1,1)
        )

        # Init params
        self._unfold_params = dict(kernel_size=(1,1), padding=(0,0))
        
    def forward(
        self,
        input: Tensor,
        weights: Tensor
    ) -> Tensor:

        """
        Arguments:
            input : input image, (B,Ci,H,W)

            weights : convolution weights, (B,Ci,Co)

        Return:
            Convolved image, input spatial dimensions are not changed (B,Co,H,W)
        """

        # Extract params
        B, Ci, H, W = input.shape
        Co = self._output_channels

        # Depthwise Convolution
        w = weights.reshape((B,Co,Ci))  # (B,Co,Ci)
        x = F.unfold(input=input, **self._unfold_params)  # (B,Ci,H,W) -> (B,Ci,HxW)
        x = torch.bmm(w, x)  # (B,Co,HxW)
        out = F.fold(input=x, output_size=(H,W), kernel_size=(1,1), padding=(0,0))  # (B,Co,HxW) -> (B,Co,H,W)
        
        return out


class SeparableGuidedConv2d(nn.Module):

    """
    A separable (factorized) guided 2D convolutional layer from the paper
    "Learning Guided Convolutional Network for Depth Completion", Tang et al., 2019
    """

    def __init__(
        self, 
        input_channels: int,
        output_channels: int,
        kernel_size: Union[int, Tuple[int,int]]
    ) -> None:

        """
        Arguments:
            input_channels : number of input channels, Ci

            output_channels : number of output channels, Co

            kernel_size : kernel size, tuple of integers, 
                          must be odd (not necessarily equal), e.g., (1,1), (3,3), (3,5), etc.
        """

        super().__init__()

        # Layers
        self._channelwise = GuidedChannelWiseConv2d(
            input_channels=input_channels,
            kernel_size=kernel_size
        )

        self._bn_relu = nn.Sequential(
            nn.BatchNorm2d(num_features=input_channels),
            nn.ReLU()
        )

        self._depthwise = GuidedDepthWiseConv2d(
            input_channels=input_channels,
            output_channels=output_channels
        )

    def forward(
        self,
        input: Tensor,
        channel_wise_weights: Tensor,
        depth_wise_weights: Tensor,
    ) -> Tensor:

        """
        Arguments:
            input : the image to be filtered, (B,Ci,H,W)

            channel_wise_weights : weights for channelwise conovlutional layer, (B,Ci,K1xK2,H,W)

            depth_wise_weights : weights for depthwise conovlutional layer, (B,Ci,Co)

        Return:
            Convolved image with the same spatial dimensions as the input image (B,Co,H,W)
        """

        x = self._channelwise(input=input, weights=channel_wise_weights)
        out = self._depthwise(input=x, weights=depth_wise_weights)

        return out


class GuidedLocalConv2d(_GuidedConvNd):
    """
    A guided local 2D convolution layer (aka Dynamic Local Filtering) from the paper
    "Dynamic Filter Networks", Brabandere et al, 2016.

    The convolutional kernels are spatially-variant and content-dependent in each site.
    """

    def __init__(
        self, 
        input_channels: int,
        output_channels: int,
        kernel_size: Union[int, Tuple[int,int]]
    ) -> None:

        """
        Arguments:
            input_channels : number of input channels, Ci

            output_channels : number of output channels, Co

            kernel_size : kernel size, tuple of integers, 
                          must be odd (not necessarily equal), e.g., (1,1), (3,3), (3,5), etc.
        """

        # Expand kernel size
        kernel_size = _expand_kernel(kernel_size, 2)

        super().__init__(        
            input_channels=input_channels,
            output_channels=output_channels,
            kernel_size=kernel_size
        )

        # Init params
        _padding = _get_padding(kernel_size)
        self._unfold_params = dict(kernel_size=kernel_size, padding=_padding)

    def forward(
        self,
        input: Tensor,
        weights: Tensor
    ) -> Tensor:

        """
        Arguments:
            input : the image to be filtered, (B,Ci,H,W)

            weights : convolution weights, (B,Ci,K1xK2,Co,H,W)

        Return:
            Convolved image with the same spatial dimensions as the input image (B,Co,H,W)
        """

        # Extract params
        B, Ci, H, W = input.shape
        Co = self._output_channels
        K = self._kernel_numel

        # Convolution
        w = weights.reshape(B,Ci*K,Co,-1).permute(0,3,1,2)  # (B,Ci,K,Co,H,W) -> (B,Ci * K,Co,HxW) -> (B,HxW,Ci x K,Co)
        x = F.unfold(input=input, **self._unfold_params).permute(0,2,1).unsqueeze(2)  # (B,Ci,H,W) -> (B,Ci x K,HxW) -> (B,HxW,1,Ci x K)
        x = torch.matmul(x, w).squeeze(2).permute(0,2,1)  # (B,HxW,1,Ci x K) x (B,HxW,Ci x K,Co) -> (B,HxW,1,Co) -> (B,HxW,Co) -> (B,Co,HxW)
        out = F.fold(input=x, output_size=(H,W), kernel_size=(1,1), padding=(0,0))  # (B,Co,H,W)

        return out


class GuidedConv2d(_GuidedConvNd):

    """
    A guided 2D convolution layer (aka Dynamic Convolutional Layer) from the paper
    "Dynamic Filter Networks", Brabandere et al, 2016.

    The convolution uses different wieghts per sample,
    weights are shared across all sites within te same sample.
    """

    def __init__(
        self, 
        input_channels: int,
        output_channels: int,
        kernel_size: Union[int, Tuple[int,int]]
    ) -> None:

        """
        Arguments:
            input_channels : number of input channels, Ci

            output_channels : number of output channels, Co

            kernel_size : kernel size, tuple of integers, 
                          must be odd (not necessarily equal), e.g., (1,1), (3,3), (3,5), etc.
        """

        # Expand kernel size
        kernel_size = _expand_kernel(kernel_size, 2)

        super().__init__(        
            input_channels=input_channels,
            output_channels=output_channels,
            kernel_size=kernel_size
        )

        # Init params
        _padding = _get_padding(kernel_size)
        self._unfold_params = dict(kernel_size=kernel_size, padding=_padding)

    def forward(
        self,
        input: Tensor,
        weights: Tensor
    ) -> Tensor:

        """
        Apply a forward step of the guided convolution

        Arguments:
            input : the image to be filtered, (B,Ci,H,W)

            weights : convolution weights, (B,Ci,K1xK2,Co)

        Return:
            Convolved image with the same spatial dimensions as the input image (B,Co,H,W)
        """

        # Extract params
        B, _, H, W = input.shape
        Co = self._output_channels

        # Convolution
        w = weights.reshape(B,-1,Co).unsqueeze(1)  # (B,Ci,K1xK2,Co) -> (B,1,Ci x K,Co)
        x = F.unfold(input=input, **self._unfold_params).permute(0,2,1).unsqueeze(2)  # (B,Ci,H,W) -> (B,Ci x K,HxW) -> (B,HxW,1,Ci x K)
        x = torch.matmul(x, w).squeeze().permute(0,2,1)  # (B,HxW,1,Ci x K) x (B,1,Ci x K,Co) -> (B,HxW,1,Co) -> (B,HxW,Co) -> (B,Co,HxW)
        out = F.fold(input=x, output_size=(H,W), kernel_size=(1,1), padding=(0,0))

        return out