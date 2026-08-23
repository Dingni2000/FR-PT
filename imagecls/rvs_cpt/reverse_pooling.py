import torch
import torch.nn.functional as F


def _pair(value):
    """Return a pooling parameter as a two-element tuple."""
    if isinstance(value, tuple):
        return value
    return (value, value)


def pooling_reverseCom_old(y, x_shape, pooling_layer):
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


def pooling_reverseCom(y, x_old, pooling_layer):
    """Project ``x_old`` onto ``{x: pooling_layer(x) == y}`` in L2 distance.

    The closed-form projection is computed independently for every non-overlapping
    pooling window.  Pixels that are not covered by a window (for example, a
    trailing row when the input size is not divisible by the kernel size) remain
    unchanged.

    For average pooling, the same offset is added to every value in a window.  For
    max pooling, values above the requested maximum are clipped; if all values are
    below it, only one of the original maxima is raised.  In particular, this does
    *not* fill a whole window with ``y``.

    Args:
        y: Desired pooling output.
        x_old: Reference input with shape ``(N, C, H, W)``.
        pooling_layer: A ``torch.nn.MaxPool2d`` or ``torch.nn.AvgPool2d`` layer.

    Returns:
        The minimum-L2-change input whose pooling output is ``y``.

    Notes:
        The closed form below requires disjoint, unpadded, undilated windows, so
        ``stride == kernel_size``, ``padding == 0``, ``ceil_mode == False`` and
        (for max pooling) ``dilation == 1`` are required.
    """
    n, channels, height, width = x_old.shape
    kernel_h, kernel_w = _pair(pooling_layer.kernel_size)

    out_h, out_w = y.shape[-2:]
    covered_h, covered_w = out_h * kernel_h, out_w * kernel_w

    # (N, C, out_h, out_w, kernel_h, kernel_w)
    windows = (
        x_old[:, :, :covered_h, :covered_w]
        .reshape(n, channels, out_h, kernel_h, out_w, kernel_w)
        .permute(0, 1, 2, 4, 3, 5)
    )
    target = y[..., None, None]

    if isinstance(pooling_layer, torch.nn.AvgPool2d):
        window_size = kernel_h * kernel_w
        divisor = pooling_layer.divisor_override or window_size
        # sum(x_new) / divisor == y, with equal changes being the L2 optimum.
        offset = (divisor * target - windows.sum(dim=(-2, -1), keepdim=True)) / window_size
        projected_windows = windows + offset
    elif isinstance(pooling_layer, torch.nn.MaxPool2d):
        clipped = torch.minimum(windows, target)
        flat_windows = windows.reshape(n, channels, out_h, out_w, -1)
        max_values, max_indices = flat_windows.max(dim=-1, keepdim=True)
        raised = flat_windows.scatter(-1, max_indices, y[..., None])
        projected_windows = torch.where(
            (max_values < y[..., None]).unsqueeze(-1),
            raised.reshape_as(windows),
            clipped,
        )
    else:  raise TypeError("pooling_layer must be torch.nn.MaxPool2d or torch.nn.AvgPool2d")

    projected_core = (
        projected_windows.permute(0, 1, 2, 4, 3, 5)
        .reshape(n, channels, covered_h, covered_w))

    projected_top = torch.cat(
        (projected_core, x_old[:, :, :covered_h, covered_w:]), dim=-1)
    return torch.cat((projected_top, x_old[:, :, covered_h:, :]), dim=-2)



if __name__ == "__main__":
    print("Testing pooling_reverseCom_new with MaxPool2d...")
    max_pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
    x_old = torch.tensor(
        [[[[5.0, 4.0, 0.0, 1.0],
           [3.0, 2.0, 2.0, 3.0],
           [8.0, 1.0, 9.0, 8.0],
           [0.0, 2.0, 7.0, 6.0]]]]
    )
    y = torch.tensor([[[[3.0, 6.0], [8.0, 7.5]]]])
    x_new = pooling_reverseCom(y, x_old, max_pool)
    torch.testing.assert_close(max_pool(x_new), y)
    print("x_old:\n", x_old)
    print("target y:\n", y)
    print("x_new:\n", x_new)
    print("MaxPool2d(x_new):\n", max_pool(x_new))

    print("\nTesting pooling_reverseCom_new with AvgPool2d...")
    avg_pool = torch.nn.AvgPool2d(kernel_size=2, stride=2)
    x_old = torch.tensor(
        [[[[1.0, 2.0, 3.0, 4.0],
           [5.0, 6.0, 7.0, 8.0],
           [9.0, 10.0, 11.0, 12.0],
           [13.0, 14.0, 15.0, 16.0]]]]
    )
    y = torch.tensor([[[[5.0, 4.0], [8.0, 12.0]]]])
    x_new = pooling_reverseCom(y, x_old, avg_pool)
    torch.testing.assert_close(avg_pool(x_new), y)
    print("x_old:\n", x_old)
    print("target y:\n", y)
    print("x_new:\n", x_new)
    print("AvgPool2d(x_new):\n", avg_pool(x_new))
    print("\nAll pooling_reverseCom_new tests passed.")
