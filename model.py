import torch
import torch.nn as nn

class ConvNext(nn.Module):
    def __init__(self,channels):
        super().__init__()

        self.depth_conv=nn.Conv2d(
            channels,
            channels,
            kernel_size=7,
            padding=(7-1)/2,
            groups=channels
        )

        self.norm=nn.LayerNorm(channels)

        # The following layers are heavily inspired from modern Transformer's MLP architecture,
        # Kernel_size = 1 replicates a linear layer pixel-wise to all input channels

        self.up_proj=nn.Conv2d(
            channels,
            4*channels,
            kernel_size=1
        )

        self.gelu=nn.GELU()

        self.down_proj=nn.Conv2d(
            channels*4,
            channels,
            kernel_size=1
        )

    def forward(self,img):
        # img.shape = B x C X H X W
        out=self.depth_conv(img)

        out.permute(0,2,3,1) # (B,H,W,C)
        self.norm(out)
        out.permute(0,3,1,2) # (B,C,H,W)

        out=self.up_proj(out) # (B,4*C,H,W)
        out=self.gelu(out)
        out=self.down_proj(out) # (B,4*C,H,W)

        out=out+img

        return out
    
class DownSampler(nn.Module):
    def __init__(self,channels):
        super().__init__()

        self.conv=nn.Conv2d(
            channels,
            2*channels,
            kernel_size=2,
            stride=2
        )

    def forward(self,img):
        return self.conv(img) #(B,2*C,H/2,W/2)
    
