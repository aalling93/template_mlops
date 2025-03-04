import torch
import torch.nn as nn
from .MODEL_CONFIG import DEFAULT_ACTIVATION
from typing import Union




def autopad(kernel_size, padding=None, dilation=1):
    """Calculate padding for 'same' shape outputs in convolution.

    This function calculates the padding required to achieve 'same' shape outputs
    in convolutional operations given the kernel size, padding, and dilation.

    Args:
        kernel_size (int or list of ints): The size of the convolutional kernel.
            If an int is provided, it's assumed to be a square kernel.
            If a list of ints is provided, each dimension's kernel size is specified.
        padding (int or list of ints, optional): The desired padding.
            If not provided, it's calculated automatically for 'same' shape outputs.
        dilation (int, optional): The dilation factor. Defaults to 1.

    Returns:
        int or list of ints: The calculated padding for 'same' shape outputs.

    Raises:
        ValueError: If the provided kernel_size is not an integer or a list of integers.

    Examples:
        >>> autopad(3)
        1
    """
    if dilation > 1:
        kernel_size = dilation * (kernel_size - 1) + 1 if isinstance(kernel_size, int) else [dilation * (x - 1) + 1 for x in kernel_size]

    if padding is None:
        padding = kernel_size // 2 if isinstance(kernel_size, int) else [x // 2 for x in kernel_size]

    return padding


class CustomConv(nn.Module):
    """Custom convolution layer with various options.

    This class represents a customizable 2D convolutional layer with options
    for kernel size, stride, padding, groups, dilation, and activation function.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int or tuple, optional): Size of the convolutional kernel.
            Defaults to 1.
        stride (int or tuple, optional): Stride for the convolution operation.
            Defaults to 1.
        padding (int or tuple, optional): Padding for the convolution operation.
            If not provided, it's calculated for 'same' shape outputs.
        groups (int, optional): Number of groups for grouped convolution. Defaults to 1.
        dilation (int, optional): Dilation factor for the convolution operation.
            Defaults to 1.
        activation (bool or nn.Module, optional): Activation function to apply after
            convolution. Defaults to True, which uses the SiLU activation function.

    Attributes:
        default_activation (nn.Module): Default activation function (SiLU).

    Methods:
        forward(x): Apply convolution, batch normalization, and activation to input tensor.
        forward_fuse(x): Perform transposed convolution of 2D data.

    Examples:
        >>> conv_layer = CustomConv2D(64, 128, kernel_size=3, stride=1, padding=1, activation=True)
        >>> input_tensor = torch.randn(1, 64, 32, 32)
        >>> output = conv_layer(input_tensor)

    Attributes:
        default_activation (nn.Module): Default activation function (SiLU).
    """

    # Default activation (SiLU)
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        groups=1,
        dilation=1,
        activation=True,
    ):
        """
        Initialize the CustomConv2D layer with given arguments.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int or tuple, optional): Size of the convolutional kernel.
                Defaults to 1.
            stride (int or tuple, optional): Stride for the convolution operation.
                Defaults to 1.
            padding (int or tuple, optional): Padding for the convolution operation.
                If not provided, it's calculated for 'same' shape outputs.
            groups (int, optional): Number of groups for grouped convolution. Defaults to 1.
            dilation (int, optional): Dilation factor for the convolution operation.
                Defaults to 1.
            activation (bool or nn.Module, optional): Activation function to apply after
                convolution. Defaults to True, which uses the SiLU activation function.

        Raises:
            ValueError: If activation argument is not a valid boolean or nn.Module.
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            autopad(kernel_size, padding, dilation),
            groups=groups,
            dilation=dilation,
            bias=False,
        )

        self.default_activation = DEFAULT_ACTIVATION  # nn.SiLU()

        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = self.default_activation if activation is True else activation if isinstance(activation, nn.Module) else nn.Identity()

    def forward(self, x):
        """
        Apply convolution, batch normalization, and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.activation(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """
        Perform transposed convolution of 2D data.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.activation(self.conv(x))


class Bottleneck(nn.Module):
    """Standard bottleneck module for deep neural networks."""

    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        groups=1,
        kernels=(3, 3),
        expansion=0.5,
    ):
        """
        Initializes a bottleneck module with the specified input and output channels, shortcut option, group,
        kernels, and expansion.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            shortcut (bool): Whether to use shortcut connections (default: True).
            groups (int): Number of groups for group convolution (default: 1).
            kernels (tuple): Tuple of kernel sizes for the two convolution layers (default: (3, 3)).
            expansion (float): Expansion factor for hidden channels (default: 0.5).
        """
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # Hidden channels
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernels[0], stride=1, padding=1, groups=1)
        self.conv2 = nn.Conv2d(
            hidden_channels,
            out_channels,
            kernels[1],
            stride=1,
            padding=1,
            groups=groups,
        )
        self.shortcut = shortcut and in_channels == out_channels

    def forward(self, x):
        """
        Forward pass for the bottleneck module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        if self.shortcut:
            return x + self.conv2(self.conv1(x))
        else:
            return self.conv2(self.conv1(x))


class SPPF(nn.Module):
    """
    Spatial Pyramid Pooling - Fast (SPPF) layer.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        kernel_sizes (tuple): Tuple of kernel sizes for max-pooling (default: (5, 9, 13)).

    The SPPF layer is a crucial component in object detection models. It plays a
    vital role in capturing multi-scale features from the input feature map, which is essential
    for accurately detecting objects of different sizes within an image.

    The layer works by performing max-pooling with multiple kernel sizes and then concatenating
    the results. This process allows the network to focus on objects at various scales, from small
    to large, in a single forward pass.

    The SPPF layer consists of the following components:
    - The first convolutional layer (conv1) reduces the number of input channels while preserving
      important features.
    - Max-pooling is applied with multiple kernel sizes specified in kernel_sizes. This pooling
      operation captures features at different receptive field sizes, enabling the detection of both
      very small objects and large objects.
    - The final convolutional layer (conv2) combines the information from all the pooled feature maps
      to produce the output feature map.

    This design enables the model to detect objects of various sizes, including very small objects
    (e.g., 3 pixels) and large objects (e.g., 300 pixels), and improves its ability to localize and
    classify objects accurately within an image.
    """

    def __init__(self, in_channels, out_channels, kernel_sizes=(3, 5, 7)):
        """
        Initializes the SPPF layer with given input/output channels and kernel sizes.

        Args:
            in_channels (int): Input channels.
            out_channels (int): Output channels.
            kernel_sizes (tuple): Tuple of kernel sizes for max-pooling (default: (5, 9, 13)).
        """
        super().__init__()
        hidden_channels = in_channels // 2  # Calculate the number of hidden channels
        self.conv1 = CustomConv(in_channels, hidden_channels, 1, 1)  # First convolution layer
        # Create max-pooling layers with custom kernel sizes specified in kernel_sizes
        self.max_pooling_layers = nn.ModuleList([nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2) for k in kernel_sizes])
        self.conv2 = CustomConv(hidden_channels * (len(kernel_sizes) + 1), out_channels, 1, 1)  # Second convolution layer

    def forward(self, x):
        """
        Forward pass through the SPPF layer.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        x = self.conv1(x)  # Apply the first convolution
        pooled_feature_maps = [self.max_pooling_layers[i](x) for i in range(len(self.max_pooling_layers))]
        # Concatenate the original input and pooled feature maps
        concatenated_features = torch.cat([x] + pooled_feature_maps, 1)
        return self.conv2(concatenated_features)


class BottleneckCSP(nn.Module):
    """CSP Bottleneck module based on https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(
        self,
        in_channels,
        out_channels,
        num_blocks=1,
        shortcut=True,
        groups=1,
        expansion=0.5,
    ):
        """
        Initializes the CSP Bottleneck module with the specified input and output channels, number of blocks,
        shortcut option, groups, and expansion factor.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            num_blocks (int): Number of bottleneck blocks in the module (default: 1).
            shortcut (bool): Whether to use shortcut connections (default: True).
            groups (int): Number of groups for group convolution (default: 1).
            expansion (float): Expansion factor for hidden channels (default: 0.5).
        """
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # Hidden channels
        self.conv1 = CustomConv(in_channels, hidden_channels, 1, 1)
        self.conv2 = nn.Conv2d(in_channels, hidden_channels, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, 1, 1, bias=False)
        self.conv4 = CustomConv(2 * hidden_channels, out_channels, 1, 1)
        self.batch_norm = nn.BatchNorm2d(2 * hidden_channels)  # Applied to the concatenation of conv2 and conv3
        self.activation = nn.SiLU()
        self.bottleneck_blocks = nn.Sequential(
            *(Bottleneck(hidden_channels, hidden_channels, shortcut, groups, expansion=1.0) for _ in range(num_blocks))
        )

    def forward(self, x):
        """
        Applies the CSP Bottleneck module with 3 convolutions.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        y1 = self.conv3(self.bottleneck_blocks(self.conv1(x)))
        y2 = self.conv2(x)
        concatenated = torch.cat((y1, y2), 1)
        return self.conv4(self.activation(self.batch_norm(concatenated)))
