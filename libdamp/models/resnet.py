"""ResNet implementation

All of this code is copied from torchvision.models.resnet. Only the input size of the first layer of the resnet
 architecture is changes to allow for a 2D input
"""

import torch
from torch import Tensor, nn


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """Create a 3x3 convolutional layer with padding.

    This is a standard building block for ResNet, applying convolution with
    kernel size 3x3 and padding to preserve spatial dimensions when stride=1.

    Parameters
    ----------
    in_planes : int
        Number of input channels
    out_planes : int
        Number of output channels
    stride : int
        Stride for the convolution (default: 1)
    groups : int
        Number of groups for grouped convolution (default: 1)
    dilation : int
        Dilation rate for dilated convolution (default: 1)

    Returns
    -------
    nn.Conv2d
        A Conv2d module configured as a 3x3 convolution
    """
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """Create a 1x1 convolutional layer.

    Used in ResNet for bottleneck layers to change the number of channels
    with minimal computational cost. Can also be used for spatial downsampling
    when stride > 1.

    Parameters
    ----------
    in_planes : int
        Number of input channels
    out_planes : int
        Number of output channels
    stride : int
        Stride for the convolution (default: 1)

    Returns
    -------
    nn.Conv2d
        A Conv2d module configured as a 1x1 convolution
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """ResNet BasicBlock with two 3x3 convolutions.

    This is the building block for ResNet-18 and ResNet-34. It consists of two
    consecutive 3x3 convolutional layers with batch normalization and ReLU
    activation, plus a residual connection (skip connection) around the block.
    """

    expansion: int = 1
    """Factor by which the output channels are expanded relative to input (always 1 for BasicBlock)."""

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample=None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer=None,
    ) -> None:
        """Initialize a ResNet BasicBlock.

        Parameters
        ----------
        inplanes : int
            Number of input channels
        planes : int
            Number of output channels for the main path
        stride : int
            Stride for the first convolution (default: 1)
        downsample : nn.Module, optional
            Module applied to the input to match output channels and spatial dimensions
        groups : int
            Number of groups for grouped convolution (default: 1). Must be 1 for BasicBlock.
        base_width : int
            Base width (default: 64). Must be 64 for BasicBlock.
        dilation : int
            Dilation rate (default: 1). Must be 1 for BasicBlock.
        norm_layer : type, optional
            Normalization layer class (default: BatchNorm2d)

        Raises
        ------
        ValueError
            If groups != 1 or base_width != 64
        NotImplementedError
            If dilation > 1
        """
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        input_channels=1,
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        self.groups = groups
        self.base_width = width_per_group
        #################################################################################
        self.conv1 = nn.Conv2d(input_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        # This is different from the orgiginal pytorch code, variable input channel size
        #################################################################################
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    block,
    layers,
    **kwargs,
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    return model


def resnet18(**kwargs) -> ResNet:
    """ResNet-18 from `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`__.

    Parameters
    ----------
    **kwargs
        Parameters passed to the `torchvision.models.resnet.ResNet` base class. Please refer to the
        `source code <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
        for more details about this class.
    """
    return _resnet(BasicBlock, [2, 2, 2, 2], **kwargs)
