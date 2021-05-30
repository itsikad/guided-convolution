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
```
clone https://github.com/itsikad/guided-convolution.git
cd guided-convolution
python setup.py install
```

# How To Use
Layers can be easily integrated into existing architectures by replacing `nn.Conv2D` layers with:

```
B, Ci, H, W = (16,3,28,28)  # batch size, input channels, height, width
Co = 5  # output channels

input = torch.randn((B,Ci,H,W))

dynamic_conv = GuidedLocalConv2d(
    input_channels=Ci,
    output_channels=Co,
    kernel_size=(3,3)
    )

weights = torch.randn((B,Ci,3*3,Co,H,W))
out = dynamic_conv(input, weights)
```

OR

```
input = torch.randn((B,Ci,H,W))

sep_conv = SeparableGuidedConv2d(
    input_channels=Ci,
    output_channels=Co,
    kernel_size=(3,3)
    )

cw_weights = torch.randn((B,Ci,3*3,H,W))  # channelwise convolution weights
dw_weights = torch.randn((B,Ci,Co))  # depthwise convolution weights
out = sep_conv(input, cw_weights, dw_weights)
```

# TODO:
1. Improve example
2. Add dilation, stride, padding etc.
3. Add hyperfan-in/out initialization presented in [O. Chang, L Flokas, H. Lipson. Principled Weight Initialization For Hypernetworks. In _Internatinoal Conference on Learning Representation (ICLR). 2020_](https://openreview.net/pdf?id=H1lma24tPB)