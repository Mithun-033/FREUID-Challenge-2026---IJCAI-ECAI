"""Model definitions for a ConvNeXt-style regression network.

This module defines a small ConvNeXt-like backbone implemented in PyTorch
including lightweight normalization and downsampling helpers. It constructs
`Model` which produces a single scalar regression output from an input image.

Classes:
 - `config`: dataclass holding network width and depth hyperparameters.
 - `layernorm`: channel-wise LayerNorm wrapper that supports NCHW tensors.
 - `ConvNext`: a ConvNeXt-inspired residual block (depthwise conv + MLP).
 - `DownSampler`: reduces spatial resolution and doubles channel dimension.
 - `Model`: full network composing stem, stages and classification head.
"""

import torch
import torch.nn as nn
from torchinfo import summary
from Config import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class layernorm(nn.Module):
    """Channel-wise Layer Normalization for NCHW tensors.

    PyTorch's `nn.LayerNorm` expects channels last (e.g. shape [..., C]).
    This wrapper permutes an incoming tensor from NCHW to NHWC, applies
    `LayerNorm`, then permutes the result back to NCHW so it can be used
    inside convolutional blocks.

    Args:
        channels: Number of channels (C) in the input tensor.
    """

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        """Apply layer normalization over the channel dimension.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Tensor of same shape as input with channel-wise normalized values.
        """
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x

class GRN(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self,x):
        gx = torch.norm(x, p = 2, dim = (-1, -2), keepdim = True)  # returns sqrt(a^2 + b^2 + c^2 ....) for all values in H, W.
        nx = gx/(gx.mean(dim = 1, keepdim = True) + 1e-6) # relative signal 

        return self.gamma * (x * nx) + x + self.beta
    


class ConvNext(nn.Module):
    """A ConvNeXt-style residual block using depthwise conv + MLP.

    The block applies a depthwise convolution for spatial mixing followed by
    a channel-wise MLP implemented as two `1x1` convolutions separated by a
    GELU non-linearity. A residual connection adds the input to the output.

    Args:
        channels: Number of input and output channels for the block.
    """

    def __init__(self, channels, config):
        super().__init__()

        self.depth_conv = nn.Conv2d(
            channels, channels, kernel_size=7, padding=(7 - 1) // 2, groups=channels
        )
        self.norm = layernorm(channels)

        # The following layers are heavily inspired from modern Transformer's MLP architecture,
        # Kernel_size = 1 replicates a linear layer pixel-wise to all input channels

        self.up_proj = nn.Conv2d(
            channels, config.expansion_ratio * channels, kernel_size=1
        )

        self.gelu = nn.GELU()
        self.grn = GRN(channels * config.expansion_ratio)
        self.down_proj = nn.Conv2d(
            channels * config.expansion_ratio, channels, kernel_size=1
        )

        # stable early training, block adds ~ 0 to the output initally ...
        self.layerscale = nn.Parameter(torch.ones((channels, 1, 1)) * config.layer_scale_init_value)  

    def forward(self, img):
        """Forward pass of the ConvNext block.

        Args:
            img: Input tensor of shape (B, C, H, W).

        Returns:
            Output tensor of the same shape as `img`, produced by the
            depthwise conv -> norm -> MLP -> residual add pipeline.
        """
        # img.shape = B x C X H X W
        out = self.depth_conv(img)

        out = self.norm(out)

        out = self.up_proj(out)  # (B,6*C,H,W)
        out = self.gelu(out)
        out = self.grn(out)
        out = self.down_proj(out)  # (B,C,H,W)

        out = self.layerscale * out + img

        return out


class DownSampler(nn.Module):
    """Downsamples spatial resolution while increasing channels.

    `DownSampler` first applies channel-wise normalization then a 2x2
    stride-2 convolution which halves the height/width and doubles the
    channel count.

    Args:
        channels: Number of input channels.
    """

    def __init__(self, channels):
        super().__init__()

        self.norm = layernorm(channels)
        self.conv = nn.Conv2d(channels, 2 * channels, kernel_size=2, stride=2)

    def forward(self, img):
        """Apply normalization then strided convolution.

        Args:
            img: Input tensor of shape (B, C, H, W).

        Returns:
            Tensor of shape (B, 2*C, H//2, W//2).
        """
        return self.conv(self.norm(img))  # (B,2*C,H/2,W/2)


class Model(nn.Module):
    """Full ConvNeXt-like model that outputs a single scalar per input.

    The network is composed of a small convolutional stem, four stages of
    `ConvNext` blocks separated by `DownSampler`s, a global average pooling
    and a small MLP head producing a single regression output.

    Args:
        config: Instance of the `config` dataclass containing widths and
            block counts for each stage.
    """

    def __init__(self, config):
        super().__init__()

        # (B,3,H,W)
        self.stem = nn.Conv2d(
            in_channels=3, out_channels=config.dim1, kernel_size=4, stride=4
        )

        # (B,96,H,W)
        self.block1 = nn.ModuleList(
            [ConvNext(config.dim1, config) for _ in range(config.num_block1)]
        )
        self.down_sample_1 = DownSampler(config.dim1)

        # (B,192,H//2,W//2)
        self.block2 = nn.ModuleList(
            [ConvNext(config.dim2, config) for _ in range(config.num_block2)]
        )
        self.down_sample_2 = DownSampler(config.dim2)

        # (B,384,H//4,W//4)
        self.block3 = nn.ModuleList(
            [ConvNext(config.dim3, config) for _ in range(config.num_block3)]
        )
        self.down_sample_3 = DownSampler(config.dim3)

        # (B,768,H//8,W//4)
        self.block4 = nn.ModuleList(
            [ConvNext(config.dim4, config) for _ in range(config.num_block4)]
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # (B,768,1,1)
        self.Linear = nn.Sequential(nn.Linear(config.dim4,256), nn.GELU(), nn.Linear(256, 1))

    def forward(self, img):
        """Forward pass for the full model.

        Args:
            img: Input image tensor of shape (B, 3, H, W).

        Returns:
            Regression output tensor of shape (B, 1).
        """
        x = self.stem(img)

        for block in self.block1:
            x = block(x)
        x = self.down_sample_1(x)

        for block in self.block2:
            x = block(x)
        x = self.down_sample_2(x)

        for block in self.block3:
            x = block(x)
        x = self.down_sample_3(x)

        for block in self.block4:
            x = block(x)

        x = self.global_pool(x)

        x = x.squeeze(-1).squeeze(-1)

        x = self.Linear(x)

        return x


if __name__ == "__main__":
    print("Loading model...")
    model = Model(config()).to(device)
    print("Model Loaded!")

    # model=torch.compile(model, mode = "max-autotune", fullgraph = False).to(device)

    print("Generating summary...")
    summary(model, input_size=(1, 3, 224, 224))
