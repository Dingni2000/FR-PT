import torch
import torch.nn.functional as F

def pooling_reverseCom(y, x_shape, pooling_layer):
    """
    Args:
        y: Pooling output with shape (batch, channels, h', w').
        x_shape: Shape of the original input x.
        pool_layer: nn.MaxPool2d layer.
    
    Returns:
        x_reconstructed: Tensor with the same spatial size as x.
    """
    assert pooling_layer.kernel_size == pooling_layer.stride, "Only support stride == kernel_size"
    if x_shape[-1] % pooling_layer.kernel_size != 0 or x_shape[-2] % pooling_layer.kernel_size != 0:
        print("[WARNING] The input size of Pool is not a multiple of the kernel size, which may introduce errors.")
    
    return F.interpolate(
        y,
        size=x_shape[-2:],
        mode='nearest',
        align_corners=None
    )
