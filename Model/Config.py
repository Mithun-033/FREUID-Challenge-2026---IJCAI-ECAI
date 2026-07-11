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

    dim1 : int = 108
    num_block1 : int = 4

    dim2 : int = dim1*2
    num_block2 : int = 2

    dim3 : int = dim1*4
    num_block3 : int = 9
    
    dim4 : int = dim1*8
    num_block4 : int = 3

    expansion_ratio : int = 4
    layer_scale_init_value : float = 1e-1
    linear_dim : int = 512
    patch_size : int = 4

@dataclass 
class train_config:
    """Hyper-parameter container for training parameters.

    Attributes:
        lr: Learning rate for the optimizer.
        weight_decay: Weight decay for the optimizer.
    """
    lr : float = 2.5e-4
    weight_decay : float = 5e-2
    min_lr : float = lr * 0.1

@dataclass
class data_config:
    image_dim : int = 352
    batch_size : int = 16
    num_workers : int = 4
    prefetch_factor : int = 2