import torch
import torch.nn.functional as F

"""
forward: activate(old_z) == old_a
reverse: activate(target_z) == target_a
         min ||target_z - old_z||_2^2

These helpers compute a feasible pre-activation target and choose the
solution closest to old_z when an activation has multiple inverse images.
"""

def _act_name(act):
    if isinstance(act, str):
        return act.lower().replace("_", "")
    if isinstance(act, type):
        return act.__name__.lower().replace("_", "")
    if isinstance(act, torch.nn.Module):
        return act.__class__.__name__.lower().replace("_", "")
    if hasattr(act, "__name__"):
        return act.__name__.lower().replace("_", "")
    return act.__class__.__name__.lower().replace("_", "")


def route_rvs_act(act, old_z: torch.Tensor, target_a: torch.Tensor, **kwargs):
    """
    Route an activation object/function/name to its reverse-computation helper.

    `act` can be a torch function such as `torch.relu`, a functional API such
    as `F.relu`, an nn.Module instance such as `nn.ReLU()`, an activation class
    such as `nn.ReLU`, or a string such as `"relu"`.
    """
    name = _act_name(act)

    if name == "relu":
        return rev_relu(old_z, target_a)
    if name == "relu6":
        return rev_relu6(old_z, target_a)
    if name == "leakyrelu":
        negative_slope = kwargs.get("negative_slope", getattr(act, "negative_slope", 0.01))
        return rev_leaky_relu(old_z, target_a, negative_slope=negative_slope)
    if name == "hardtanh":
        min_val = kwargs.get("min_val", getattr(act, "min_val", -1.0))
        max_val = kwargs.get("max_val", getattr(act, "max_val", 1.0))
        return rev_hardtanh(old_z, target_a, min_val=min_val, max_val=max_val)
    if name == "tanh":
        return rev_tanh(old_z, target_a, eps=kwargs.get("eps", 1e-6))
    if name == "sigmoid":
        return rev_sigmoid(old_z, target_a, eps=kwargs.get("eps", 1e-6))
    if name == "hardsigmoid":
        return rev_hardsigmoid(old_z, target_a)
    if name == "elu":
        alpha = kwargs.get("alpha", getattr(act, "alpha", 1.0))
        return rev_elu(old_z, target_a, alpha=alpha, eps=kwargs.get("eps", 1e-6))
    if name == "softplus":
        beta = kwargs.get("beta", getattr(act, "beta", 1.0))
        return rev_softplus(old_z, target_a, beta=beta, eps=kwargs.get("eps", 1e-6))
    if name == "gelu":
        return rev_gelu(old_z, target_a, iters=kwargs.get("iters", 80))
    if name in ("silu", "swish"):
        return rev_silu(old_z, target_a, iters=kwargs.get("iters", 80))
    if name == "hardswish":
        return rev_hardswish(old_z, target_a)
    if name == "softmax":
        dim = kwargs.get("dim", getattr(act, "dim", -1))
        return rev_softmax(old_z, target_a, dim=dim, eps=kwargs.get("eps", 1e-12))

    raise NotImplementedError(f"Unsupported activation for reverse computation: {act!r}")


def _check_same_shape(old_z: torch.Tensor, target_a: torch.Tensor):
    assert old_z.shape == target_a.shape, (
        f"old_z.shape={old_z.shape}, target_a.shape={target_a.shape}"
    )


def rev_relu(old_z: torch.Tensor, target_a: torch.Tensor):
    _check_same_shape(old_z, target_a)
    target_z = torch.where((target_a > 0), target_a, torch.clamp(old_z, max=0))
    return target_z


def rev_relu6(old_z: torch.Tensor, target_a: torch.Tensor):
    """
    ReLU6:
        a = min(max(z, 0), 6)

    At the saturation boundaries, choose the feasible value closest to old_z.
    Targets outside [0, 6] are projected to the valid output range.
    """
    _check_same_shape(old_z, target_a)
    safe_a = target_a.clamp(min=0, max=6)
    return torch.where(
        safe_a <= 0,
        torch.clamp(old_z, max=0),
        torch.where(safe_a >= 6, torch.clamp(old_z, min=6), safe_a),
    )


def rev_leaky_relu(old_z: torch.Tensor, target_a: torch.Tensor, negative_slope: float = 0.01):
    """
    LeakyReLU:
        a = z,                 z >= 0
        a = negative_slope*z,  z < 0

    For a positive slope the function is strictly monotonic and can be
    inverted directly. A zero slope delegates to the ReLU inverse.
    """
    _check_same_shape(old_z, target_a)
    assert negative_slope >= 0, "negative_slope should be non-negative."
    if negative_slope == 0:
        return rev_relu(old_z, target_a)
    return torch.where(target_a >= 0, target_a, target_a/0.5)


def rev_hardtanh(
    old_z: torch.Tensor,
    target_a: torch.Tensor,
    min_val: float = -1.0,
    max_val: float = 1.0,
):
    """
    HardTanh:
        a = clamp(z, min_val, max_val)

    Saturated targets have multiple inverse images; choose the feasible value
    closest to old_z.
    """
    _check_same_shape(old_z, target_a)
    if min_val >= max_val:
        raise ValueError("min_val should be smaller than max_val.")

    safe_a = target_a.clamp(min=min_val, max=max_val)
    return torch.where(
        safe_a <= min_val,
        torch.clamp(old_z, max=min_val),
        torch.where(safe_a >= max_val, torch.clamp(old_z, min=max_val), safe_a),
    )


def rev_tanh(old_z: torch.Tensor, target_a: torch.Tensor, eps: float = 1e-6):
    """
    Use atanh after clamping the target to a finite part of (-1, 1).
    """
    _check_same_shape(old_z, target_a)
    target_a = target_a.clamp(min=-1 + eps, max=1 - eps)
    return torch.atanh(target_a)


def rev_sigmoid(old_z: torch.Tensor, target_a: torch.Tensor, eps: float = 1e-6):
    """
    Use logit after clamping the target to a finite part of (0, 1).
    """
    _check_same_shape(old_z, target_a)
    target_a = target_a.clamp(min=eps, max=1 - eps)
    return torch.logit(target_a)


def rev_hardsigmoid(old_z: torch.Tensor, target_a: torch.Tensor):
    """
    PyTorch HardSigmoid:
        a = 0,           z <= -3
        a = z / 6 + 0.5, -3 < z < 3
        a = 1,           z >= 3

    At the saturation boundaries, choose the feasible value closest to old_z.
    """
    _check_same_shape(old_z, target_a)
    safe_a = target_a.clamp(min=0, max=1)
    middle_z = 6 * safe_a - 3
    return torch.where(
        safe_a <= 0,
        torch.clamp(old_z, max=-3),
        torch.where(safe_a >= 1, torch.clamp(old_z, min=3), middle_z),
    )


def rev_elu(old_z: torch.Tensor, target_a: torch.Tensor, alpha: float = 1.0, eps: float = 1e-6):
    """
    ELU:
        a = z,                    z > 0
        a = alpha * (exp(z)-1),   z <= 0

    For positive alpha, invert the positive and negative branches directly.
    Targets below the finite range are clamped near -alpha.
    """
    _check_same_shape(old_z, target_a)
    if alpha <= 0:
        raise ValueError("alpha should be positive.")

    lower = -alpha + eps
    safe_a = target_a.clamp(min=lower)
    return torch.where(safe_a > 0, safe_a, torch.log1p(safe_a / alpha))


def rev_softplus(old_z: torch.Tensor, target_a: torch.Tensor, beta: float = 1.0, eps: float = 1e-6):
    """
    Softplus:
        a = log(1 + exp(beta*z)) / beta

    Use a numerically stable form of log(expm1(beta * a)) / beta.
    Targets at zero are clamped to a finite approximation.
    """
    _check_same_shape(old_z, target_a)
    if beta <= 0:
        raise ValueError("beta should be positive.")

    safe_a = target_a.clamp(min=eps)
    beta_a = beta * safe_a
    # Rewrite log(expm1(x)) to avoid overflow for large x.
    return (beta_a + torch.log1p(-torch.exp(-beta_a))) / beta


def _normal_cdf(x: torch.Tensor):
    return 0.5 * (1.0 + torch.erf(x / torch.sqrt(torch.tensor(2.0, device=x.device, dtype=x.dtype))))


def _gelu_exact(x: torch.Tensor):
    return x * _normal_cdf(x)


def rev_gelu(old_z: torch.Tensor, target_a: torch.Tensor, iters: int = 80):
    """
    Solve the exact GELU inverse with bisection on its monotonic branches.
    When two roots exist, choose the one closest to old_z.
    """
    _check_same_shape(old_z, target_a)

    z_min = torch.tensor(-0.7517915246935645, device=target_a.device, dtype=target_a.dtype)
    a_min = _gelu_exact(z_min)

    target_a = target_a.clamp(min=a_min)
    flat_a = target_a.reshape(-1)

    # Right root. Negative targets use [z_min, 0]; non-negative targets use
    # [0, hi], where hi is expanded from the largest target.
    hi_value = max(float(flat_a.max().detach().cpu()) + 8.0, 8.0)
    lo_right = torch.where(flat_a < 0, z_min.expand_as(flat_a), torch.zeros_like(flat_a))
    hi_right = torch.full_like(flat_a, hi_value)
    for _ in range(iters):
        mid = (lo_right + hi_right) / 2
        val = _gelu_exact(mid)
        lo_right = torch.where(val < flat_a, mid, lo_right)
        hi_right = torch.where(val >= flat_a, mid, hi_right)
    right_root = (lo_right + hi_right) / 2

    # Left root for negative targets. The finite lower bound is sufficient
    # because GELU(-30) is effectively zero in floating point.
    lo_left = torch.full_like(flat_a, -30.0)
    hi_left = z_min.expand_as(flat_a)
    for _ in range(iters):
        mid = (lo_left + hi_left) / 2
        val = _gelu_exact(mid)
        # GELU decreases as mid increases on the left branch.
        lo_left = torch.where(val > flat_a, mid, lo_left)
        hi_left = torch.where(val <= flat_a, mid, hi_left)
    left_root = (lo_left + hi_left) / 2

    old_flat = old_z.reshape(-1)
    choose_left = (flat_a < 0) & ((left_root - old_flat).abs() < (right_root - old_flat).abs())
    result = torch.where(choose_left, left_root, right_root)
    return result.reshape_as(target_a)


def _silu(x: torch.Tensor):
    return x * torch.sigmoid(x)


def rev_silu(old_z: torch.Tensor, target_a: torch.Tensor, iters: int = 80):
    """
    SiLU / Swish:
        silu(z) = z * sigmoid(z)

    SiLU is non-monotonic on the negative branch. Use bisection and choose
    the inverse root closest to old_z when two roots exist.
    """
    _check_same_shape(old_z, target_a)

    z_min = torch.tensor(-1.2784645427610738, device=target_a.device, dtype=target_a.dtype)
    a_min = _silu(z_min)
    target_a = target_a.clamp(min=a_min)
    flat_a = target_a.reshape(-1)

    hi_value = max(float(flat_a.max().detach().cpu()) + 8.0, 8.0)
    lo_right = torch.where(flat_a < 0, z_min.expand_as(flat_a), torch.zeros_like(flat_a))
    hi_right = torch.full_like(flat_a, hi_value)
    for _ in range(iters):
        mid = (lo_right + hi_right) / 2
        val = _silu(mid)
        lo_right = torch.where(val < flat_a, mid, lo_right)
        hi_right = torch.where(val >= flat_a, mid, hi_right)
    right_root = (lo_right + hi_right) / 2

    lo_left = torch.full_like(flat_a, -60.0)
    hi_left = z_min.expand_as(flat_a)
    for _ in range(iters):
        mid = (lo_left + hi_left) / 2
        val = _silu(mid)
        lo_left = torch.where(val > flat_a, mid, lo_left)
        hi_left = torch.where(val <= flat_a, mid, hi_left)
    left_root = (lo_left + hi_left) / 2

    old_flat = old_z.reshape(-1)
    choose_left = (flat_a < 0) & ((left_root - old_flat).abs() < (right_root - old_flat).abs())
    result = torch.where(choose_left, left_root, right_root)
    return result.reshape_as(target_a)


def rev_hardswish(old_z: torch.Tensor, target_a: torch.Tensor):
    """
    PyTorch HardSwish:
        a = z * ReLU6(z + 3) / 6

    The piecewise form is:
        z <= -3:      a = 0
        -3 < z < 3:  a = z * (z + 3) / 6
        z >= 3:      a = z

    The middle segment is a quadratic equation:
        z^2 + 3z - 6a = 0
        z = (-3 +/- sqrt(9 + 24a)) / 2

    HardSwish is not one-to-one in the middle range. Clamp unreachable targets
    and choose the closest valid root.
    """
    _check_same_shape(old_z, target_a)
    safe_a = target_a.clamp(min=-0.375)
    disc = torch.clamp(9 + 24 * safe_a, min=0)
    sqrt_disc = torch.sqrt(disc)
    root_left = (-3 - sqrt_disc) / 2
    root_right = (-3 + sqrt_disc) / 2

    dist_left = (root_left - old_z).abs()
    dist_right = (root_right - old_z).abs()
    middle_root = torch.where(dist_left < dist_right, root_left, root_right)

    zero_root = torch.where(
        torch.clamp(old_z, max=-3).sub(old_z).abs() < old_z.abs(),
        torch.clamp(old_z, max=-3),
        torch.zeros_like(old_z),
    )

    return torch.where(
        safe_a < 0,
        middle_root,
        torch.where(
            safe_a == 0,
            zero_root,
            torch.where(safe_a < 3, root_right, safe_a),
        ),
    )


def rev_softmax(old_z: torch.Tensor, target_a: torch.Tensor, dim: int = -1, eps: float = 1e-12):
    """
    Softmax:
        softmax(z_i) = exp(z_i) / sum_j exp(z_j)

    For a valid probability distribution, every inverse has the form:
        z_i = log(target_a_i) + c
    where c is a shared offset along the softmax dimension. The closest
    solution uses:
        c = mean_i(old_z_i - log(p_i))

    Negative targets are rejected. Zero probabilities are clamped and
    renormalized to produce finite logits.
    """
    _check_same_shape(old_z, target_a)
    if torch.any(target_a < 0):
        raise ValueError("target_a for softmax should be non-negative.")
    if torch.any(target_a.sum(dim=dim, keepdim=True) <= 0):
        raise ValueError("target_a should have positive sum along softmax dim.")

    probs = target_a.clamp(min=eps)
    probs = probs / probs.sum(dim=dim, keepdim=True)
    base_logits = torch.log(probs)
    c = (old_z - base_logits).mean(dim=dim, keepdim=True)
    return base_logits + c
