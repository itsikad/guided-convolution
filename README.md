# Guided Convolutions
A PyTorch implementations of guided convolutions presented in:

1. [B. De Brabandere, X. Jia, T. Tuytelaars, and L. Van Gool. Dynamic filter networks. In _Neural Information
Processing Systems (NIPS). 2016_](https://arxiv.org/pdf/1605.09673.pdf)
    1. Dynamic Local Filtering (GuidedLocalConv2d)
    2. Dynamic Convolutional Layer (GuidedConv2d)

2. [J. Tang, F. Tian, W. Feng, J. Li and P. Tan. Learning Guided Convolutional Network for Depth Completion. In _IEEE TRANSACTIONS ON IMAGE PROCESSING (TIP). 2019_](https://arxiv.org/pdf/1908.01238.pdf)
    1. Channelwise 2D Guided Convolution (GuidedChannelWiseConv2d)
    2. Depthwise 2D Guided convolution (GuidedDepthWiseConv2d)
    3. Separable/Factorized Guided Convolution (SeparableGuidedConv2d)

# How To Use
Layers can be easily integrated into existing architectures by replacing `nn.Conv2D` layers with:
```
conv = DynamicLocalFilteringLayer(
    input_channels=16,
    output_channels=64,
    kernel_size=(3,3)
    )
```
OR
```
conv = SeparableGuidedConvLayer(
    input_channels=16,
    output_channels=64,
    kernel_size=(3,3)
    )
```

# TODO:
1. Improve example
2. Add dilation, stride, padding etc.
3. Add hyperfan-in/out initialization presented in [O. Chang, L Flokas, H. Lipson. Principled Weight Initialization For Hypernetworks. In _Internatinoal Conference on Learning Representation (ICLR). 2020_](https://openreview.net/pdf?id=H1lma24tPB)