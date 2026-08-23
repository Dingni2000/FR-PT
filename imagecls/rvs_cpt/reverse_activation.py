import torch
import torch.nn.functional as F

"""
forward: activate(old_z) == old_a
reverse: activate(target_z) == target_a
         min ||target_z - old_z||_2^2

这里的“反向计算”不是反向传播求梯度，而是：
给定一个期望的激活值 target_a，求一个新的 pre-activation target_z，
使 activate(target_z) 尽量等于 target_a，并且在所有可行 target_z 中
选择离 old_z 最近的那个。

不同 activation 的可逆性不同：
1. tanh / sigmoid / leaky ReLU 这类单调函数可以直接写解析逆。
2. ReLU 在 target_a == 0 时有无穷多个原像 z <= 0，需要选离 old_z 最近的点。
3. GELU / SiLU 在负半轴不是一一映射，部分 target_a 可能对应两个 z，
   需要分别求根后选择离 old_z 最近的那个。
4. Softmax 的输出对 logits 加同一个常数不变，因此反解是一整条平移族，
   需要选择离 old_z 最近的那组 logits。
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

    它在 a==0 和 a==6 时都不是一一映射：
    1. target_a == 0 时，任意 z <= 0 都可行，选 min(old_z, 0)。
    2. 0 < target_a < 6 时，只能 z = target_a。
    3. target_a == 6 时，任意 z >= 6 都可行，选 max(old_z, 6)。

    若 target_a 超出 [0, 6]，不存在精确反解，这里先投影到合法值域。
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

    当 negative_slope > 0 时它是严格单调函数，所以可以直接反解：
        z = a,                         a >= 0
        z = a / negative_slope,        a < 0

    如果 negative_slope == 0，它退化成普通 ReLU。
    """
    _check_same_shape(old_z, target_a)
    assert negative_slope >= 0, "negative_slope should be non-negative."
    if negative_slope == 0:
        return rev_relu(old_z, target_a)
    # return torch.where(target_a >= 0, target_a, target_a / negative_slope) # TODO 容易放大target_a造成数值爆炸
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

    饱和区间有无穷多个原像：
    1. target_a == min_val 时，任意 z <= min_val 都可行，选 min(old_z, min_val)。
    2. min_val < target_a < max_val 时，z = target_a。
    3. target_a == max_val 时，任意 z >= max_val 都可行，选 max(old_z, max_val)。
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
    tanh 的值域是 (-1, 1)，解析逆为 atanh(a)。

    如果 target_a 精确等于 -1 或 1，数学上需要 z 为无穷大；
    为了得到有限 tensor，这里先 clamp 到 (-1+eps, 1-eps)。
    """
    _check_same_shape(old_z, target_a)
    target_a = target_a.clamp(min=-1 + eps, max=1 - eps)
    return torch.atanh(target_a)


def rev_sigmoid(old_z: torch.Tensor, target_a: torch.Tensor, eps: float = 1e-6):
    """
    sigmoid 的值域是 (0, 1)，解析逆为 logit(a)=log(a/(1-a))。

    target_a 等于 0 或 1 时反解是无穷大；这里 clamp 后返回有限近似。
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

    因此：
    1. target_a == 0 时，选 min(old_z, -3)。
    2. 0 < target_a < 1 时，z = 6 * target_a - 3。
    3. target_a == 1 时，选 max(old_z, 3)。
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

    当 alpha > 0 时可单调反解：
        z = a,                    a > 0
        z = log(a/alpha + 1),     -alpha < a <= 0

    ELU 的负半轴值域下界是 -alpha。若 target_a <= -alpha，
    精确反解需要 z -> -inf；这里 clamp 到 -alpha+eps 后给出有限近似。
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

    它是严格单调函数，解析逆为：
        z = log(exp(beta*a)-1) / beta

    当 target_a == 0 时需要 z -> -inf；这里 clamp 到 eps 返回有限近似。
    使用 log(expm1(.)) 可以比 log(exp(.)-1) 更稳定。
    """
    _check_same_shape(old_z, target_a)
    if beta <= 0:
        raise ValueError("beta should be positive.")

    safe_a = target_a.clamp(min=eps)
    beta_a = beta * safe_a
    # log(exp(x)-1) = x + log(1-exp(-x))，大 x 时不会因为 exp(x) 溢出。
    return (beta_a + torch.log1p(-torch.exp(-beta_a))) / beta


def _normal_cdf(x: torch.Tensor):
    return 0.5 * (1.0 + torch.erf(x / torch.sqrt(torch.tensor(2.0, device=x.device, dtype=x.dtype))))


def _gelu_exact(x: torch.Tensor):
    return x * _normal_cdf(x)


def rev_gelu(old_z: torch.Tensor, target_a: torch.Tensor, iters: int = 80):
    """
    GELU 精确形式：
        gelu(z) = z * Phi(z)
    其中 Phi 是标准正态分布的 CDF。

    GELU 的难点：
    1. 它不是全局单调函数，负半轴附近会先下降再上升。
    2. 它的最小值约为 -0.16997，对应 z 约为 -0.75179。
    3. target_a 在 (min_gelu, 0) 内通常有两个原像：
       一个在 (-inf, z_min)，另一个在 (z_min, 0)。

    处理办法：
    1. 对 target_a < min_gelu 的值，不存在精确反解，投影到 GELU 的最小点。
    2. 对 target_a >= 0，GELU 在 [0, +inf) 上单调，二分求唯一非负根。
    3. 对 min_gelu <= target_a < 0，同时在两个单调区间二分求两个根，
       再选离 old_z 更近的那个，符合 min ||target_z-old_z||_2^2。
    """
    _check_same_shape(old_z, target_a)

    z_min = torch.tensor(-0.7517915246935645, device=target_a.device, dtype=target_a.dtype)
    a_min = _gelu_exact(z_min)

    target_a = target_a.clamp(min=a_min)
    flat_a = target_a.reshape(-1)

    # 右侧根：区间 [z_min, hi]。target_a < 0 时右根落在 [z_min, 0]；
    # target_a >= 0 时右根落在 [0, hi]。hi 会按最大 target_a 自适应放大。
    hi_value = max(float(flat_a.max().detach().cpu()) + 8.0, 8.0)
    lo_right = torch.where(flat_a < 0, z_min.expand_as(flat_a), torch.zeros_like(flat_a))
    hi_right = torch.full_like(flat_a, hi_value)
    for _ in range(iters):
        mid = (lo_right + hi_right) / 2
        val = _gelu_exact(mid)
        lo_right = torch.where(val < flat_a, mid, lo_right)
        hi_right = torch.where(val >= flat_a, mid, hi_right)
    right_root = (lo_right + hi_right) / 2

    # 左侧根：只对 target_a < 0 有意义，区间 [lo, z_min] 单调递减。
    # 对非常接近 0 的负数，左根会很靠近 -inf；这里用 -30 作为数值下界，
    # 在 float32/float64 中 GELU(-30) 已经几乎等于 0。
    lo_left = torch.full_like(flat_a, -30.0)
    hi_left = z_min.expand_as(flat_a)
    for _ in range(iters):
        mid = (lo_left + hi_left) / 2
        val = _gelu_exact(mid)
        # 在左侧区间，gelu(mid) 随 mid 增大而减小：
        # val > target 说明 mid 太靠左，需要右移；否则需要左移。
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

    它和 GELU 类似，不是全局一一映射，最小值约为 -0.27846。
    target_a 在 (min_silu, 0) 内有两个原像；这里同样用二分分别求两个根，
    再选离 old_z 最近的那个。
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

    分段写出来是：
        z <= -3:      a = 0
        -3 < z < 3:  a = z * (z + 3) / 6
        z >= 3:      a = z

    中间段是二次方程：
        z^2 + 3z - 6a = 0
        z = (-3 +/- sqrt(9 + 24a)) / 2

    HardSwish 也不是一一映射：
    1. target_a 在 (-3/8, 0) 内，中间段有两个原像，选离 old_z 最近的。
    2. target_a == 0 时，z <= -3 都可行，同时 z=0 也可行，比较后选最近。
    3. target_a >= 3 时，右侧线性段 z=target_a。
    4. target_a < -3/8 不可达，投影到最小点 z=-1.5。
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

    若 target_a 是合法概率分布，所有反解都满足：
        z_i = log(target_a_i) + c
    其中 c 是同一条 softmax 维度上的任意常数，因为 softmax(z+c)=softmax(z)。

    为了满足 min ||target_z-old_z||_2^2，需要求最优平移 c：
        minimize sum_i (log(p_i)+c-old_z_i)^2
    对 c 求导并令其为 0：
        c = mean_i(old_z_i - log(p_i))

    注意：
    1. target_a 中不能有负数。
    2. 若 target_a 有 0，log(0)=-inf，没有有限 logits 能精确产生 0 概率；
       这里用 eps clamp 并重新归一化，返回有限近似。
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


def _max_abs_err(x: torch.Tensor, y: torch.Tensor):
    return (x - y).abs().max().item()


if __name__ == "__main__":
    torch.manual_seed(0)

    old_z = torch.tensor([[-2.0, -0.4, 0.0, 0.8, 3.0]])

    relu_a = torch.tensor([[0.0, 0.0, 0.0, 1.2, 4.0]])
    relu_z = rev_relu(old_z, relu_a)
    print("ReLU target_z:", relu_z)
    print("ReLU max error:", _max_abs_err(F.relu(relu_z), relu_a))

    relu6_a = F.relu6(torch.tensor([[-2.0, 0.0, 2.5, 6.0, 9.0]]))
    relu6_z = rev_relu6(old_z, relu6_a)
    print("\nReLU6 target_z:", relu6_z)
    print("ReLU6 max error:", _max_abs_err(F.relu6(relu6_z), relu6_a))

    leaky_a = F.leaky_relu(torch.tensor([[-3.0, -0.5, 0.0, 2.0]]), negative_slope=0.1)
    leaky_z = rev_leaky_relu(old_z[:, :4], leaky_a, negative_slope=0.1)
    print("\nLeakyReLU target_z:", leaky_z)
    print("LeakyReLU max error:", _max_abs_err(F.leaky_relu(leaky_z, negative_slope=0.1), leaky_a))

    hardtanh_a = F.hardtanh(torch.tensor([[-2.0, -0.4, 0.0, 0.8, 2.0]]))
    hardtanh_z = rev_hardtanh(old_z, hardtanh_a)
    print("\nHardTanh target_z:", hardtanh_z)
    print("HardTanh max error:", _max_abs_err(F.hardtanh(hardtanh_z), hardtanh_a))

    tanh_a = torch.tanh(torch.tensor([[-2.0, -0.5, 0.0, 0.5, 2.0]]))
    tanh_z = rev_tanh(old_z, tanh_a)
    print("\nTanh target_z:", tanh_z)
    print("Tanh max error:", _max_abs_err(torch.tanh(tanh_z), tanh_a))

    sigmoid_a = torch.sigmoid(torch.tensor([[-3.0, -1.0, 0.0, 1.0, 3.0]]))
    sigmoid_z = rev_sigmoid(old_z, sigmoid_a)
    print("\nSigmoid target_z:", sigmoid_z)
    print("Sigmoid max error:", _max_abs_err(torch.sigmoid(sigmoid_z), sigmoid_a))

    hardsigmoid_a = F.hardsigmoid(torch.tensor([[-5.0, -1.0, 0.0, 2.0, 5.0]]))
    hardsigmoid_z = rev_hardsigmoid(old_z, hardsigmoid_a)
    print("\nHardSigmoid target_z:", hardsigmoid_z)
    print("HardSigmoid max error:", _max_abs_err(F.hardsigmoid(hardsigmoid_z), hardsigmoid_a))

    elu_a = F.elu(torch.tensor([[-2.0, -0.3, 0.0, 1.5]]), alpha=1.0)
    elu_z = rev_elu(old_z[:, :4], elu_a, alpha=1.0)
    print("\nELU target_z:", elu_z)
    print("ELU max error:", _max_abs_err(F.elu(elu_z, alpha=1.0), elu_a))

    softplus_a = F.softplus(torch.tensor([[-4.0, -1.0, 0.0, 1.0, 4.0]]))
    softplus_z = rev_softplus(old_z, softplus_a)
    print("\nSoftplus target_z:", softplus_z)
    print("Softplus max error:", _max_abs_err(F.softplus(softplus_z), softplus_a))

    gelu_source_z = torch.tensor([[-3.0, -1.0, -0.2, 0.0, 2.0]])
    gelu_a = F.gelu(gelu_source_z, approximate="none")
    gelu_z = rev_gelu(old_z, gelu_a)
    print("\nGELU target_z:", gelu_z)
    print("GELU max error:", _max_abs_err(F.gelu(gelu_z, approximate="none"), gelu_a))

    silu_source_z = torch.tensor([[-4.0, -1.5, -0.2, 0.0, 2.0]])
    silu_a = F.silu(silu_source_z)
    silu_z = rev_silu(old_z, silu_a)
    print("\nSiLU target_z:", silu_z)
    print("SiLU max error:", _max_abs_err(F.silu(silu_z), silu_a))

    hardswish_source_z = torch.tensor([[-4.0, -2.0, -0.5, 0.0, 2.0]])
    hardswish_a = F.hardswish(hardswish_source_z)
    hardswish_z = rev_hardswish(old_z, hardswish_a)
    print("\nHardSwish target_z:", hardswish_z)
    print("HardSwish max error:", _max_abs_err(F.hardswish(hardswish_z), hardswish_a))

    softmax_old_z = torch.tensor([[2.0, -1.0, 0.5], [-0.3, 1.2, 0.0]])
    softmax_target_a = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    softmax_z = rev_softmax(softmax_old_z, softmax_target_a, dim=-1)
    print("\nSoftmax target_z:", softmax_z)
    print("Softmax max error:", _max_abs_err(F.softmax(softmax_z, dim=-1), softmax_target_a))
