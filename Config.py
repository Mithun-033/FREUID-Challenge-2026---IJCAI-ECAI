from dataclasses import dataclass

@dataclass
class config:
    """Hyper-parameter container for network widths and depths.

    Attributes:
        dim1: Number of channels in the first stage.
        num_block1: Number of `ConvNext` blocks in the first stage.
        dim2: Number of channels in the second stage.
        num_block2: Number of `ConvNext` blocks in the second stage.
        dim3: Number of channels in the third stage.
        num_block3: Number of `ConvNext` blocks in the third stage.
        dim4: Number of channels in the fourth stage.
        num_block4: Number of `ConvNext` blocks in the fourth stage.
    """

    dim1 : int = 96
    num_block1 : int = 3

    dim2 : int = 96*2
    num_block2 : int = 3

    dim3 : int = 96*4
    num_block3 : int = 9
    
    dim4 : int = 96*8
    num_block4 : int = 3