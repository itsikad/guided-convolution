# Guided Convolutions
A PyTorch implementations of guided convolutions presented in:

1. [B. De Brabandere, X. Jia, T. Tuytelaars, and L. Van Gool. Dynamic filter networks. In _Neural Information
Processing Systems (NIPS). 2016_](https://arxiv.org/pdf/1605.09673.pdf)
    1. Dynamic Local Filtering (GuidedLocalConv2d)
    2. Dynamic Convolutional (GuidedConv2d)

2. [J. Tang, F. Tian, W. Feng, J. Li and P. Tan. Learning Guided Convolutional Network for Depth Completion. In _IEEE TRANSACTIONS ON IMAGE PROCESSING (TIP). 2019_](https://arxiv.org/pdf/1908.01238.pdf)
    1. Channelwise 2D Guided Convolution (GuidedChannelWiseConv2d)
    2. Depthwise 2D Guided convolution (GuidedDepthWiseConv2d)
    3. Separable/Factorized Guided Convolution (SeparableGuidedConv2d)

# Setup
Add to an existing project using pip
```
pip install -e git+https://github.com/itsikad/guided-convolution.git#egg=guided_conv
```

Clone and install this repository

```
clone https://github.com/itsikad/guided-convolution.git
cd guided-convolution
python setup.py install
```

# How To Use
Layers can be easily integrated into existing architectures by replacing `nn.Conv2D` 
and adding kernel generation networks (hypernet).

## Code example
Imports and initialization

```
import torch
import torch.nn as nn
from guided_conv import GuidedLocalConv2d, SeparableGuidedConv2d

B, Ci, H, W = (16,3,28,28)  # batch size, input channels, height, width
Co = 5  # output channels

input = torch.randn((B,Ci,H,W))
```

Dynamic local filtering layer:

```
dynamic_conv = GuidedLocalConv2d(
    input_channels=Ci,
    output_channels=Co,
    kernel_size=(3,3)
    )

# Kernel (weights) Generation Netowrk
kgn = nn.Conv2d(
    in_channels=Ci,
    out_channels=Ci*3*3*Co,
    kernel_size=(3,3),
    padding=(1,1)
    )
weights = kgn(input).reshape((B,Ci,3*3,Co,H,W))

out = dynamic_conv(input, weights)
```

Separable Guided Convolution:

```
sep_conv = SeparableGuidedConv2d(
    input_channels=Ci,
    output_channels=Co,
    kernel_size=(3,3)
    )

# Kernel (weights) Generation Netowrk (channelwise and depthwise weights)
cw_kgn = nn.Conv2d(
    in_channels=Ci,
    out_channels=Ci*3*3,
    kernel_size=(3,3),
    padding=(1,1)
    )

dw_kgn = nn.Sequential(
    nn.Conv2d(
        in_channels=Ci,
        out_channels=Ci*Co,
        kernel_size=(3,3),
        padding=(1,1)
        ),
    nn.AdaptiveAvgPool2d(output_size=(1,1))
    )

cw_weights = cw_kgn(input).reshape((B,Ci,3*3,H,W))  # channelwise convolution weights
dw_weights = dw_kgn(input).reshape((B,Ci,Co))  # depthwise convolution weights

out = sep_conv(input, cw_weights, dw_weights)
```

# TODO:
1. Improve example
2. Add dilation, stride, padding etc.
3. Add hyperfan-in/out initialization presented in [O. Chang, L Flokas, H. Lipson. Principled Weight Initialization For Hypernetworks. In _Internatinoal Conference on Learning Representation (ICLR). 2020_](https://openreview.net/pdf?id=H1lma24tPB)