import math
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from utils import (
        check_fft_reliablity,
        check_reliability,
        mdp_tikhonov_fft_solve,
        remove_bias,
    )
except ImportError:
    from .utils import (
        check_fft_reliablity,
        check_reliability,
        mdp_tikhonov_fft_solve,
        remove_bias,
    )


def _batch_flat_norm(x):
    return torch.linalg.norm(x.reshape(x.shape[0], -1), dim=-1)


def _cache(_layer_params):
    cache = getattr(_layer_params, "_frpt_reverse_cache", None)
    if cache is None:
        cache = {}
        _layer_params._frpt_reverse_cache = cache
    return cache


def _cache_key(_layer_params, tag, *args):
    w = _layer_params.weight
    return (tag, *args, w.dtype, w.device, int(w._version))


def build_conv2d_matrix(_layer_params, input_shape):
    """Build the exact linear map used by torch conv2d for one sample."""
    _, c1, h1, w1 = input_shape
    weight = _layer_params.weight.detach()
    eye = torch.eye(c1 * h1 * w1, dtype=weight.dtype, device=weight.device)
    basis = eye.reshape(c1 * h1 * w1, c1, h1, w1)
    out = F.conv2d(
        basis, weight, bias=None, stride=_layer_params.stride,
        padding=_layer_params.padding, dilation=_layer_params.dilation,
        groups=_layer_params.groups,
    )
    return out.reshape(c1 * h1 * w1, -1).T.contiguous()


def _exact_expand_normal_solve(A, target, x0, regu, kappa_eff=1e3, max_abs=1e3):
    residual = target - x0 @ A.mT
    normal = A.mT @ A
    rhs = A.mT @ residual.mT
    if regu:
        eps = torch.finfo(normal.real.dtype).eps
        sigma_sq_upper = torch.diagonal(normal).real.sum().clamp_min(eps)
        alpha = sigma_sq_upper / float(kappa_eff) ** 2
        margin = max_abs - x0.abs().amax(dim=-1)
        bound = (torch.linalg.vector_norm(residual, dim=-1) / (2 * margin.clamp_min(eps))).square().amax()
        normal.diagonal().add_(torch.maximum(alpha, bound))
    return x0 + torch.linalg.solve(normal, rhs).mT


def convo_reverseCom_exact_shrink(front_fea, back_fea_star, _layer_params, regu=True):
    bs = front_fea.shape[0]
    A = build_conv2d_matrix(_layer_params, front_fea.shape)
    x0 = front_fea.detach().reshape(bs, -1)
    target = remove_bias(back_fea_star.detach(), _layer_params).reshape(bs, -1)
    residual = target - x0 @ A.T

    gram = A @ A.T
    lagrange = torch.linalg.solve(gram, residual.T)
    x = x0 + (A.T @ lagrange).T
    reliable = check_reliability(x, A, target, "solve", x_ref=x0)
    if (not regu) or reliable.all():
        return x.reshape_as(front_fea)

    failed = ~reliable
    eps = torch.finfo(gram.real.dtype).eps
    alpha = torch.diagonal(gram).real.sum().clamp_min(eps) / 1e6
    failed_residual, failed_x0 = residual[failed], x0[failed]
    margin = 1e3 - failed_x0.abs().amax(dim=-1)
    bound = (torch.linalg.vector_norm(failed_residual, dim=-1) / (2 * margin.clamp_min(eps))).square().amax()
    gram.diagonal().add_(torch.maximum(alpha, bound))
    lagrange = torch.linalg.solve(gram, failed_residual.T)
    x[failed] = failed_x0 + (A.T @ lagrange).T
    return x.reshape_as(front_fea)


def convo_reverseCom_exact_expand(front_fea, back_fea_star, _layer_params, regu=True):
    bs = front_fea.shape[0]
    A = build_conv2d_matrix(_layer_params, front_fea.shape)
    x0 = front_fea.detach().reshape(bs, -1)
    target = remove_bias(back_fea_star.detach(), _layer_params).reshape(bs, -1)
    try:
        x = torch.linalg.lstsq(A, target.T).solution.T
    except RuntimeError:
        x = _exact_expand_normal_solve(A, target, x0, regu=regu)
    reliable = check_reliability(x, A, target, "lstsq", x_ref=x0)
    if (not regu) or reliable.all():
        return x.reshape_as(front_fea)
    failed = ~reliable
    x[failed] = _exact_expand_normal_solve(A, target[failed], x0[failed], regu=True)
    return x.reshape_as(front_fea)


def conv2d_forward_linear(x, _layer_params):
    return F.conv2d(
        x, _layer_params.weight.detach(), bias=None,
        stride=_layer_params.stride, padding=_layer_params.padding,
        dilation=_layer_params.dilation, groups=_layer_params.groups,
    )


def conv2d_adjoint_linear(y_grad, _layer_params, input_hw):
    padding = _layer_params.padding if isinstance(_layer_params.padding, tuple) else (_layer_params.padding,) * 2
    stride = _layer_params.stride if isinstance(_layer_params.stride, tuple) else (_layer_params.stride,) * 2
    dilation = _layer_params.dilation if isinstance(_layer_params.dilation, tuple) else (_layer_params.dilation,) * 2
    kh, kw = _layer_params.weight.shape[-2:]
    h_in, w_in = input_hw
    h_out, w_out = y_grad.shape[-2:]
    output_padding = (
        h_in - ((h_out - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kh - 1) + 1),
        w_in - ((w_out - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kw - 1) + 1),
    )
    return F.conv_transpose2d(
        y_grad, _layer_params.weight.detach(), bias=None, stride=stride,
        padding=padding, output_padding=output_padding,
        groups=_layer_params.groups, dilation=dilation,
    )


def estimate_conv2d_operator_norm(_layer_params, input_shape, n_iter=10):
    key = _cache_key(_layer_params, "opnorm", tuple(input_shape[1:]), n_iter)
    cache = _cache(_layer_params)
    if key in cache:
        return cache[key]

    x = torch.randn((1, *input_shape[1:]), dtype=_layer_params.weight.dtype, device=_layer_params.weight.device)
    x /= _batch_flat_norm(x).reshape(-1, 1, 1, 1).clamp_min(1e-10)
    for _ in range(n_iter):
        x = conv2d_adjoint_linear(conv2d_forward_linear(x, _layer_params), _layer_params, input_shape[-2:])
        x /= _batch_flat_norm(x).reshape(-1, 1, 1, 1).clamp_min(1e-10)
    norm = (_batch_flat_norm(conv2d_forward_linear(x, _layer_params)) / _batch_flat_norm(x).clamp_min(1e-10)).amax()
    cache[key] = norm.detach()
    return cache[key]


def check_conv_reliability(
    x_fast: torch.Tensor,
    target: torch.Tensor,
    _layer_params,
    problem: Literal["solve", "lstsq"] = "solve",
    x_ref: Optional[torch.Tensor] = None,
    A_norm: Optional[torch.Tensor] = None,
    max_abs: Optional[float] = 1e3,
    amplification_tol: float = 1e2,
) -> bool:
    x_work, target_work = x_fast.detach(), target.detach()
    if not torch.isfinite(x_work).all():
        return False
    if max_abs is not None and not (x_work.abs().amax(dim=tuple(range(1, x_work.ndim))) <= max_abs).all():
        return False

    if x_ref is not None and amplification_tol is not None:
        xf, rf = x_fast.reshape(x_fast.shape[0], -1), x_ref.reshape(x_ref.shape[0], -1)
        dn = torch.linalg.vector_norm(xf - rf, dim=1)
        rn = torch.linalg.vector_norm(rf, dim=1)
        floor = 1e-6 * math.sqrt(rf.shape[-1])
        if not torch.all((rn <= floor) | (dn <= amplification_tol * rn)):
            return False

    if A_norm is None:
        A_norm = estimate_conv2d_operator_norm(_layer_params, x_work.shape)
    tiny = torch.finfo(x_work.real.dtype).eps
    x_norm, target_norm = _batch_flat_norm(x_work), _batch_flat_norm(target_work)
    residual = conv2d_forward_linear(x_work, _layer_params) - target_work

    if problem == "solve":
        rel = _batch_flat_norm(residual) / (A_norm * x_norm + target_norm).clamp_min(tiny)
        return bool((rel <= (1e-4 if x_work.real.dtype == torch.float64 else 1e-3)).all())
    if problem == "lstsq":
        grad = conv2d_adjoint_linear(residual, _layer_params, x_work.shape[-2:])
        rel = _batch_flat_norm(grad) / (A_norm * (A_norm * x_norm + target_norm)).clamp_min(tiny)
        return bool((rel <= (1e-3 if x_work.real.dtype == torch.float64 else 1e-2)).all())
    raise ValueError(f"Unknown problem type: {problem}")


def conjugate_gradient_batched(apply_A, b, max_iter=100, tol=1e-8, damping=0.0):
    x = torch.zeros_like(b)
    r = b - (apply_A(x) + damping * x)
    p = r.clone()
    dims = tuple(range(1, b.ndim))
    rs_old = torch.sum(r * r, dim=dims)
    b_norm = torch.sqrt(torch.clamp(torch.sum(b * b, dim=dims), min=1e-10))
    for _ in range(max_iter):
        Ap = apply_A(p) + damping * p
        denom = torch.sum(p * Ap, dim=dims).clamp_min(1e-10)
        alpha = (rs_old / denom).reshape(-1, *([1] * (b.ndim - 1)))
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.sum(r * r, dim=dims)
        if torch.max(torch.sqrt(torch.clamp(rs_new, min=0.0)) / b_norm.clamp_min(1e-10)) < tol:
            break
        beta = (rs_new / rs_old.clamp_min(1e-30)).reshape(-1, *([1] * (b.ndim - 1)))
        p = r + beta * p
        rs_old = rs_new
    return x


def _cg_damping_from_norm(A_norm, kappa_eff):
    eps = torch.finfo(A_norm.real.dtype).eps
    return float((A_norm / kappa_eff).square().clamp_min(eps).item())


def _alpha_for_feature_bound(residual, anchor, max_abs=1e3):
    if max_abs is None:
        return 0.0
    margin = float(max_abs) - float(anchor.detach().abs().amax())
    if margin <= 0:
        return 0.0
    return (float(_batch_flat_norm(residual.detach()).amax()) / (2.0 * margin)) ** 2


def convo_reverseCom_cg_shrink(front_fea, back_fea_star, _layer_params, max_iter=100, tol=1e-8, regu=True, kappa_eff=1e3):
    x0 = front_fea.detach()
    target = remove_bias(back_fea_star.detach(), _layer_params)
    residual = target - conv2d_forward_linear(x0, _layer_params)
    A_norm = estimate_conv2d_operator_norm(_layer_params, x0.shape)

    def apply_AAT(alpha):
        return conv2d_forward_linear(conv2d_adjoint_linear(alpha, _layer_params, x0.shape[-2:]), _layer_params)

    beta = conjugate_gradient_batched(apply_AAT, residual, max_iter=max_iter, tol=tol)
    x = x0 + conv2d_adjoint_linear(beta, _layer_params, x0.shape[-2:])
    if (not regu) or check_conv_reliability(x, target, _layer_params, "solve", A_norm=A_norm, x_ref=x0):
        return x

    damping = max(_cg_damping_from_norm(A_norm, kappa_eff), _alpha_for_feature_bound(residual, x0))
    beta = conjugate_gradient_batched(apply_AAT, residual, max_iter=max_iter, tol=tol, damping=damping)
    return x0 + conv2d_adjoint_linear(beta, _layer_params, x0.shape[-2:])


def convo_reverseCom_cg_expand(front_fea, back_fea_star, _layer_params, max_iter=100, tol=1e-8, regu=True, kappa_eff=1e3):
    x0 = front_fea.detach()
    target = remove_bias(back_fea_star.detach(), _layer_params)
    input_hw = front_fea.shape[-2:]
    A_norm = estimate_conv2d_operator_norm(_layer_params, front_fea.shape)

    def apply_ATA(x):
        return conv2d_adjoint_linear(conv2d_forward_linear(x, _layer_params), _layer_params, input_hw)

    rhs = conv2d_adjoint_linear(target, _layer_params, input_hw)
    x = conjugate_gradient_batched(apply_ATA, rhs, max_iter=max_iter, tol=tol)
    if (not regu) or check_conv_reliability(x, target, _layer_params, "lstsq", A_norm=A_norm, x_ref=x0):
        return x

    residual = target - conv2d_forward_linear(x0, _layer_params)
    rhs = conv2d_adjoint_linear(residual, _layer_params, input_hw)
    alpha = max(_cg_damping_from_norm(A_norm, kappa_eff), _alpha_for_feature_bound(residual, x0))
    return x0 + conjugate_gradient_batched(apply_ATA, rhs, max_iter=max_iter, tol=tol, damping=alpha)


def _fft_system(_layer_params, h, w):
    key = _cache_key(_layer_params, "fft", h, w)
    cache = _cache(_layer_params)
    if key not in cache:
        weight = torch.flip(_layer_params.weight.detach(), (-2, -1))
        A = torch.fft.rfft2(weight, s=(h, w), dim=(-2, -1)).permute(2, 3, 0, 1).contiguous()
        cache[key] = {
            "A": A,
            "weight": weight,
            "weight_complex": weight.to(A.dtype),
            "a_norm": torch.linalg.matrix_norm(A.reshape(-1, *A.shape[-2:]), ord="fro"),
        }
    return cache[key]


def _frequency_factor(system, problem):
    A = system["A"].reshape(-1, *system["A"].shape[-2:])
    m, n = A.shape[-2:]
    if problem == "solve":
        name = "lu" if m == n else "gram_lu"
        if name not in system:
            M = A if m == n else A @ A.mH
            LU, pivots, info = torch.linalg.lu_factor_ex(M, check_errors=False)
            all_good = bool((info == 0).all().item())
            good_idx = None if all_good else (info == 0).nonzero(as_tuple=False).flatten()
            system[name] = (LU, pivots, all_good, good_idx)
        return system[name]
    if problem == "lstsq":
        if "qr" not in system:
            system["qr"] = torch.linalg.qr(A, mode="reduced")
        return system["qr"]
    raise ValueError(f"Unknown problem={problem}")


def _frequency_mdp_solve(
    system,
    B: torch.Tensor,
    X0: torch.Tensor,
    reliability_problem: Literal["solve", "lstsq"],
    regularize: bool = True,
):
    A = system["A"]
    freq_shape, m, n = A.shape[:-2], A.shape[-2], A.shape[-1]
    bs = B.shape[0]
    A_flat = A.reshape(-1, m, n)
    B_flat = B.reshape(bs, -1, m).transpose(0, 1).contiguous()
    X0_flat = X0.reshape(bs, -1, n).transpose(0, 1).contiguous()

    try:
        if reliability_problem == "solve":
            LU, pivots, all_good, good = _frequency_factor(system, "solve")
            if m == n:
                if all_good:
                    X_flat = torch.linalg.lu_solve(LU, pivots, B_flat.transpose(-2, -1)).transpose(-2, -1)
                else:
                    X_flat = X0_flat.clone()
                    if good.numel():
                        X_flat[good] = torch.linalg.lu_solve(
                            LU[good], pivots[good], B_flat[good].transpose(-2, -1)
                        ).transpose(-2, -1)
            else:
                residual = B_flat - torch.einsum("fmn,fbn->fbm", A_flat, X0_flat)
                X_flat = X0_flat.clone()
                idx = slice(None) if all_good else good
                if all_good or good.numel():
                    lagrange = torch.linalg.lu_solve(
                        LU[idx], pivots[idx], residual[idx].transpose(-2, -1)
                    ).transpose(-2, -1)
                    X_flat[idx] = X0_flat[idx] + torch.einsum("fmn,fbm->fbn", A_flat[idx].conj(), lagrange)
        elif reliability_problem == "lstsq":
            Q, R = _frequency_factor(system, "lstsq")
            rhs = Q.mH @ B_flat.transpose(-2, -1)
            X_flat = torch.linalg.solve_triangular(R, rhs, upper=True).transpose(-2, -1)
        else:
            raise ValueError(f"Unknown problem={reliability_problem}")
    except RuntimeError:
        X_flat = X0_flat.clone()

    if regularize:
        reliable = check_fft_reliablity(
            X_flat, A_flat, B_flat, reliability_problem, x_ref=X0_flat,
            a_norm=system["a_norm"],
        )
        failed = ~reliable
        if failed.any():
            freq_idx, batch_idx = failed.nonzero(as_tuple=True)
            unique_freq, inverse = torch.unique(freq_idx, sorted=False, return_inverse=True)
            X_reg = mdp_tikhonov_fft_solve(
                A_flat[unique_freq], B_flat[freq_idx, batch_idx], X0_flat[freq_idx, batch_idx],
                system_index=inverse,
            )[0]
            X_flat = X_flat.index_put((freq_idx, batch_idx), X_reg)

    return X_flat.transpose(0, 1).reshape(bs, *freq_shape, n)


def _boundary_terms(_layer_params, h, w):
    system = _fft_system(_layer_params, h, w)
    if "boundary" in system:
        return system

    p = _layer_params.padding[0]
    kh, kw = _layer_params.weight.shape[-2:]
    DFT_H = torch.fft.fft(torch.eye(h, dtype=_layer_params.weight.dtype, device=_layer_params.weight.device), dim=0)
    DFT_W = torch.fft.rfft(torch.eye(w, dtype=_layer_params.weight.dtype, device=_layer_params.weight.device), dim=0)
    terms = []
    for i in range(kh):
        for j in range(kw):
            xx, yy = kh - i - 1, kw - j - 1
            top, left = max(xx - p, 0), max(yy - p, 0)
            bottom, right = max(kh - 1 - xx - p, 0), max(kw - 1 - yy - p, 0)
            mask = torch.zeros((h, w), dtype=torch.bool, device=_layer_params.weight.device)
            mask[:top, :] = True
            mask[:, :left] = True
            if bottom:
                mask[-bottom:, :] = True
            if right:
                mask[:, -right:] = True
            terms.append((mask, DFT_H[:, i, None] * DFT_W[:, j]))
    system["boundary"] = terms
    return system


def convo_reverseCom_fftbound_shrink(front_fea, back_fea_star, _layer_params, regu=True):
    p = _layer_params.padding[0]
    bs, _, h, w = front_fea.shape
    kh, kw = _layer_params.weight.shape[-2:]
    back = F.pad(remove_bias(back_fea_star.detach(), _layer_params),
                 (w - back_fea_star.shape[-1], 0, h - back_fea_star.shape[-2], 0))
    fft_back = torch.fft.rfft2(torch.roll(back, shifts=(p, p), dims=(-2, -1)), dim=(-2, -1))
    fft_front = torch.fft.rfft2(front_fea, dim=(-2, -1)).permute(0, 2, 3, 1)
    system = _boundary_terms(_layer_params, h, w)

    G = torch.zeros_like(fft_back)
    if p > 0:
        for k, (mask, phase) in enumerate(system["boundary"]):
            i, j = divmod(k, kw)
            boundary = torch.fft.rfft2(front_fea * mask, dim=(-2, -1))
            G += torch.einsum("dc,bcuv->bduv", system["weight_complex"][:, :, i, j], boundary) * phase

    target = (G + fft_back).permute(0, 2, 3, 1)
    fft_res = _frequency_mdp_solve(system, target, fft_front, "solve", regularize=regu)
    return torch.fft.irfft2(fft_res.permute(0, 3, 1, 2), s=(h, w), dim=(-2, -1))


def convo_reverseCom_fftbound_expand(front_fea, back_fea_star, _layer_params, regu=True):
    p = _layer_params.padding[0]
    bs, _, h, w = front_fea.shape
    _, _, kh, kw = _layer_params.weight.shape
    back = F.pad(remove_bias(back_fea_star.detach(), _layer_params),
                 (w - back_fea_star.shape[-1], 0, h - back_fea_star.shape[-2], 0))
    fft_back = torch.fft.rfft2(torch.roll(back, shifts=(p, p), dims=(-2, -1)), dim=(-2, -1))
    fft_front = torch.fft.rfft2(front_fea, dim=(-2, -1)).permute(0, 2, 3, 1)
    system = _boundary_terms(_layer_params, h, w)

    G = torch.zeros_like(fft_back)
    if p > 0:
        for k, (mask, phase) in enumerate(system["boundary"]):
            i, j = divmod(k, kw)
            boundary = torch.fft.rfft2(front_fea * mask, dim=(-2, -1))
            G += torch.einsum("dc,bcuv->bduv", system["weight_complex"][:, :, i, j], boundary) * phase

    target = (G + fft_back).permute(0, 2, 3, 1)
    fft_res = _frequency_mdp_solve(system, target, fft_front, "lstsq", regularize=regu)
    return torch.fft.irfft2(fft_res.permute(0, 3, 1, 2), s=(h, w), dim=(-2, -1))


def convo_reverseCom_fftpadded_shrink(front_fea, back_fea_star, _layer_params, regu=True):
    p = _layer_params.padding[0]
    front = F.pad(front_fea, (p, p, p, p))
    bs, _, h, w = front.shape
    back = F.pad(remove_bias(back_fea_star.detach(), _layer_params),
                 (w - back_fea_star.shape[-1], 0, h - back_fea_star.shape[-2], 0))
    target = torch.fft.rfft2(back, dim=(-2, -1)).permute(0, 2, 3, 1)
    fft_front = torch.fft.rfft2(front, dim=(-2, -1)).permute(0, 2, 3, 1)
    system = _fft_system(_layer_params, h, w)
    fft_res = _frequency_mdp_solve(system, target, fft_front, "solve", regularize=regu)
    result = torch.fft.irfft2(fft_res.permute(0, 3, 1, 2), s=(h, w), dim=(-2, -1))
    return result[:, :, p:-p, p:-p] if p > 0 else result


def convo_reverseCom_fftpadded_expand(front_fea, back_fea_star, _layer_params, regu=True):
    p = _layer_params.padding[0]
    front = F.pad(front_fea, (p, p, p, p))
    bs, _, h, w = front.shape
    back = F.pad(remove_bias(back_fea_star.detach(), _layer_params),
                 (w - back_fea_star.shape[-1], 0, h - back_fea_star.shape[-2], 0))
    target = torch.fft.rfft2(back, dim=(-2, -1)).permute(0, 2, 3, 1)
    fft_front = torch.fft.rfft2(front, dim=(-2, -1)).permute(0, 2, 3, 1)
    system = _fft_system(_layer_params, h, w)
    fft_res = _frequency_mdp_solve(system, target, fft_front, "lstsq", regularize=regu)
    result = torch.fft.irfft2(fft_res.permute(0, 3, 1, 2), s=(h, w), dim=(-2, -1))
    return result[:, :, p:-p, p:-p] if p > 0 else result


def convo_reverseCom(front_fea, back_fea_star, _layer_params, method="cg", regu=True):
    assert _layer_params.groups == 1, "FFT path currently supports groups=1 only"
    assert _layer_params.stride == (1, 1), "FFT path currently supports stride=1 only"
    assert _layer_params.dilation == (1, 1), "FFT path currently supports dilation=1 only"

    if method == "exact_matrix":
        return (convo_reverseCom_exact_shrink if front_fea[0].numel() >= back_fea_star[0].numel()
                else convo_reverseCom_exact_expand)(front_fea, back_fea_star, _layer_params, regu=regu)
    if method == "cg":
        fn = convo_reverseCom_cg_shrink if front_fea[0].numel() >= back_fea_star[0].numel() else convo_reverseCom_cg_expand
        return fn(front_fea, back_fea_star, _layer_params, max_iter=100, tol=1e-9, regu=regu)
    if method == "fft_oldbound":
        fn = convo_reverseCom_fftbound_shrink if front_fea.shape[1] >= back_fea_star.shape[1] else convo_reverseCom_fftbound_expand
        return fn(front_fea, back_fea_star, _layer_params, regu=regu)
    if method == "fft_pad":
        fn = convo_reverseCom_fftpadded_shrink if front_fea.shape[1] >= back_fea_star.shape[1] else convo_reverseCom_fftpadded_expand
        return fn(front_fea, back_fea_star, _layer_params, regu=regu)
    raise ValueError("method must be one of: 'cg', 'fft_oldbound', 'exact_matrix', 'fft_pad'")


if __name__ == "__main__":
    torch.manual_seed(0)
    methods = ("exact_matrix", "cg", "fft_oldbound", "fft_pad")
    shrink = nn.Conv2d(2, 1, 1, bias=True)
    expand = nn.Conv2d(1, 2, 1, bias=True)
    with torch.no_grad():
        shrink.weight.copy_(torch.tensor([1.0, 2.0]).reshape(1, 2, 1, 1))
        shrink.bias.fill_(0.25)
        expand.weight.copy_(torch.tensor([1.0, 2.0]).reshape(2, 1, 1, 1))
        expand.bias.copy_(torch.tensor([0.25, -0.5]))

    cases = (
        ("shrink", shrink, torch.randn(2, 2, 4, 4)),
        ("expand", expand, torch.randn(2, 1, 4, 4)),
    )
    for case, layer, expected in cases:
        front = torch.randn_like(expected)
        target = layer(expected).detach()
        for method in methods:
            result = convo_reverseCom(front, target, layer, method=method)
            assert result.shape == front.shape and torch.isfinite(result).all()
            assert torch.allclose(layer(result), target, atol=2e-4, rtol=2e-4), (
                case, method, (layer(result) - target).abs().max().item())
            print(f"[PASS] {case:6s} / {method}")
