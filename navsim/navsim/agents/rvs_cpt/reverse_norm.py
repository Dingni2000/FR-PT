import torch
import torch.nn as nn
from typing import Optional, Tuple


def bn_reverseCom(
    x_ori: torch.Tensor,
    y: torch.Tensor,
    bn_layer: nn.Module,
    min_abs_weight: float = 1e-4,
) -> torch.Tensor:
    """Invert eval-mode BatchNorm; non-invertible affine channels return zero."""
    if not isinstance(bn_layer, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        raise TypeError(f"Expected BatchNorm1d/2d/3d, got {type(bn_layer)!r}.")
    if bn_layer.training:
        raise ValueError("bn_reverseCom requires BatchNorm in eval mode.")
    if not bn_layer.track_running_stats:
        raise ValueError("BatchNorm without running statistics is not invertible in eval mode.")
    if x_ori.shape != y.shape:
        raise ValueError(f"x_ori.shape={tuple(x_ori.shape)} does not match y.shape={tuple(y.shape)}.")
    if y.ndim < 2 or y.shape[1] != bn_layer.num_features:
        raise ValueError(f"Expected channel dimension {bn_layer.num_features}, got shape={tuple(y.shape)}.")

    shape = (1, bn_layer.num_features) + (1,) * (y.ndim - 2)
    mean = bn_layer.running_mean.detach().to(y).view(shape)
    var = bn_layer.running_var.detach().to(y).view(shape)

    if bn_layer.affine:
        weight = bn_layer.weight.detach().to(y).view(shape)
        bias = bn_layer.bias.detach().to(y).view(shape)
        valid = weight.abs() > min_abs_weight
    else:
        weight = torch.ones(shape, dtype=y.dtype, device=y.device)
        bias = torch.zeros(shape, dtype=y.dtype, device=y.device)
        valid = torch.ones(shape, dtype=torch.bool, device=y.device)

    safe_weight = torch.where(valid, weight, torch.ones_like(weight))
    recovered = (y - bias) / safe_weight * torch.sqrt(var + bn_layer.eps) + mean
    return torch.where(valid, recovered, torch.zeros_like(recovered)).to(x_ori.dtype)


def _safe_inverse_affine(
    y: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    view_shape: Tuple[int, ...],
    min_abs_weight: float = 1e-4,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Invert reliable affine entries; mark near-zero weights as non-invertible."""
    if min_abs_weight < 0:
        raise ValueError("min_abs_weight must be nonnegative.")
    if bias is not None:
        y = y - bias.detach().to(y).reshape(view_shape)
    if weight is None:
        return y, None

    weight = weight.detach().to(y).reshape(view_shape)
    valid = weight.abs() > min_abs_weight
    safe_weight = torch.where(valid, weight, torch.ones_like(weight))
    return torch.where(valid, y / safe_weight, torch.zeros_like(y)), valid


def _reverse_normalized_tensor(
    normalized_target: torch.Tensor,
    reduce_dims: Tuple[int, ...],
    front_fea: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
    min_var_gap: float = 1e-6,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Reverse z=(x-mean(x))/sqrt(var(x)+eps).

    Known normalized entries are preserved whenever feasible. Entries whose
    affine weights are non-invertible are completed with the minimum-norm
    values satisfying mean(z)=0. The free LayerNorm shift is then chosen so
    those non-invertible input positions are exactly zero.
    """
    if front_fea is not None and front_fea.shape != normalized_target.shape:
        raise ValueError(
            f"front_fea.shape={tuple(front_fea.shape)} does not match "
            f"normalized_target.shape={tuple(normalized_target.shape)}."
        )

    reduce_dims = tuple(dim % normalized_target.dim() for dim in reduce_dims)
    z = normalized_target

    if valid_mask is None:
        invalid = None
        z = z - z.mean(dim=reduce_dims, keepdim=True)
    else:
        valid = valid_mask.expand_as(z)
        invalid = ~valid
        n_invalid = invalid.sum(dim=reduce_dims, keepdim=True)
        known_sum = torch.where(valid, z, torch.zeros_like(z)).sum(
            dim=reduce_dims, keepdim=True)
        fill = -known_sum / n_invalid.clamp_min(1)
        completed = torch.where(valid, z, fill)
        centered = z - z.mean(dim=reduce_dims, keepdim=True)
        z = torch.where(n_invalid > 0, completed, centered)

    q = torch.mean(z.square(), dim=reduce_dims, keepdim=True)

    # Project only infeasible targets onto the numerically usable zero-mean ball.
    max_q = 1.0 - min_var_gap
    shrink = torch.sqrt(
        torch.clamp(max_q / q.clamp_min(torch.finfo(z.dtype).tiny),max=1.0))
    z = z * shrink
    q = torch.mean(z.square(), dim=reduce_dims, keepdim=True)

    scale = torch.sqrt(
        torch.as_tensor(eps, device=z.device, dtype=z.dtype)/ (1.0 - q).clamp_min(min_var_gap))
    centered = scale * z

    if invalid is None:
        if front_fea is None:
            mean = torch.zeros_like(centered.mean(dim=reduce_dims, keepdim=True))
        else:
            mean = (front_fea.to(z) - centered).mean(dim=reduce_dims, keepdim=True)
    else:
        n_invalid = invalid.sum(dim=reduce_dims, keepdim=True)
        zero_invalid_mean = -torch.where(invalid, centered, torch.zeros_like(centered)
            ).sum(dim=reduce_dims, keepdim=True) / n_invalid.clamp_min(1)

        if front_fea is None:
            fallback_mean = torch.zeros_like(zero_invalid_mean)
        else:
            fallback_mean = (front_fea.to(z) - centered).mean(dim=reduce_dims, keepdim=True)
        mean = torch.where(n_invalid > 0, zero_invalid_mean, fallback_mean)

    return mean + centered


def layer_norm_reverseCom(
    back_fea: torch.Tensor,
    ln_layer: nn.LayerNorm,
    front_fea: Optional[torch.Tensor] = None,
    min_abs_weight: float = 1e-12,
) -> torch.Tensor:
    """Reverse LayerNorm with stable handling of zero/near-zero affine weights."""
    if not isinstance(ln_layer, nn.LayerNorm):
        raise TypeError(f"Expected nn.LayerNorm, got {type(ln_layer)!r}.")

    normalized_shape = tuple(ln_layer.normalized_shape)
    if back_fea.dim() < len(normalized_shape):
        raise ValueError(
            f"back_fea has {back_fea.dim()} dims, normalized_shape={normalized_shape}.")
    if tuple(back_fea.shape[-len(normalized_shape):]) != normalized_shape:
        raise ValueError(
            f"Expected trailing dims {normalized_shape}, got shape={tuple(back_fea.shape)}.")

    view_shape = (1,) * (back_fea.dim() - len(normalized_shape)) + normalized_shape
    normalized_target, valid_mask = _safe_inverse_affine(
        back_fea,
        ln_layer.weight if ln_layer.elementwise_affine else None,
        ln_layer.bias if ln_layer.elementwise_affine else None,
        view_shape,
        min_abs_weight,
    )
    reduce_dims = tuple(range(back_fea.dim() - len(normalized_shape), back_fea.dim()))
    return _reverse_normalized_tensor(
        normalized_target,
        reduce_dims,
        front_fea,
        ln_layer.eps,
        valid_mask=valid_mask,
    )
