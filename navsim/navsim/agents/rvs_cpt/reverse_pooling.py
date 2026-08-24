import torch
import torch.nn.functional as F

def pooling_reverseCom(y, x_shape, pooling_layer):
    """
    Args:
        y: Pooling 的输出，shape (batch, channels, h', w')
        x_shape: 原始输入 x 的 shape
        pool_layer: nn.MaxPool2d 层
    
    Returns:
        x_reconstructed: 与原始输入 x 相同大小的张量
    """
    assert pooling_layer.kernel_size == pooling_layer.stride, "Only support stride == kernel_size"
    if x_shape[-1] % pooling_layer.kernel_size != 0 or x_shape[-2] % pooling_layer.kernel_size != 0:
        print("[WARNING] The input size of Pool is not a multiple of the kernel size, which may introduce errors.")
    
    return F.interpolate(
        y,
        size=x_shape[-2:],
        mode='nearest',  # 最近邻插值
        align_corners=None
    )


if __name__ == "__main__":
    print("Testing pooling_reverseCom...")
    pooling_layer = torch.nn.MaxPool2d(kernel_size=2, stride=2)
    x = torch.randn(1, 2, 4, 4)
    y = pooling_layer(x)
    x_reconstructed = pooling_reverseCom(y, x.shape, pooling_layer)
    print("Original x:\n", x)
    print("Reconstructed x:\n", x_reconstructed)


