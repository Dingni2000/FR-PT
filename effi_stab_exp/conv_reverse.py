import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal, Dict, Tuple

def _real_dtype_of(dtype):
    if dtype in (torch.complex64, torch.float16, torch.bfloat16, torch.float32):
        return torch.float32
    return torch.float64

def _default_tol(dtype):
    real_dtype = _real_dtype_of(dtype)
    return 1e-4 if real_dtype == torch.float32 else 1e-8

def _batch_flat_norm(x):
    return torch.linalg.norm(x.reshape(x.shape[0], -1), dim=-1)

def check_reliability(
    x_fast: torch.Tensor,
    A: torch.Tensor,
    b: torch.Tensor,
    problem: Literal["solve", "lstsq"] = "solve",
    max_abs: Optional[float] = 1e2,
    max_norm_amp: Optional[float] = 1e4,
    residual_tol: Optional[float] = 0.1,
    opt_tol: Optional[float] = 0.1,
    return_mask: bool = False,
):
    """
    Matrix reliability check for row-wise solutions.

    A is [m, n], b is [bs, m], x_fast is [bs, n]. The forward model is
    b ~= x @ A.T. Complex systems intentionally use transpose in the
    forward residual and conjugation only in the least-squares gradient.
    """
    assert x_fast.dim() == 2, f"Expected x_fast to have shape [bs,n], got {x_fast.shape}"
    assert A is not None and A.dim() == 2, f"Expected A to have shape [m,n], got {None if A is None else A.shape}"
    assert b is not None and b.dim() == 2 and b.shape[0] == x_fast.shape[0] and b.shape[1] == A.shape[0], \
        f"Expected b to have shape [bs,m], got {None if b is None else b.shape}"

    bs = x_fast.shape[0]
    real_dtype = _real_dtype_of(x_fast.dtype)
    device = x_fast.device
    tiny = torch.tensor(torch.finfo(real_dtype).eps, dtype=real_dtype, device=device)
    reliable = torch.ones(bs, dtype=torch.bool, device=device)

    finite_mask = torch.isfinite(x_fast).reshape(bs, -1).all(dim=-1)
    if not finite_mask.all():
        print('[reliable] got non-finite values')
        reliable &= finite_mask

    if max_abs is not None:
        x_abs_max = x_fast.abs().amax(dim=-1)
        abs_mask = x_abs_max <= max_abs
        if not abs_mask.all():
            print(f'[reliable] some value exceeds {max_abs}')
            reliable &= abs_mask

    x_norm = torch.linalg.norm(x_fast, dim=-1).to(real_dtype)
    A_norm = torch.linalg.norm(A, ord="fro").to(device=device, dtype=real_dtype).expand(bs)
    b_norm = torch.linalg.norm(b, dim=-1).to(real_dtype)

    if max_norm_amp is not None:
        norm_amp = (A_norm * x_norm) / b_norm.clamp_min(tiny)
        amp_mask = norm_amp <= max_norm_amp
        if not amp_mask.all():
            print(f"[reliable] ||A|| ||x|| / ||b|| > {max_norm_amp}")
            reliable &= amp_mask

    residual = x_fast @ A.mT - b

    if problem == "solve":
        tol = _default_tol(x_fast.dtype) if residual_tol is None else residual_tol
        denom = A_norm * x_norm + b_norm
        residual_norm = torch.linalg.norm(residual, dim=-1).to(real_dtype)
        rel_backward_error = residual_norm / denom.clamp_min(tiny)
        res_mask = rel_backward_error <= tol
        if not res_mask.all():
            print(f"[reliable solve] ||Ax-b|| / (||A|| ||x|| + ||b||) > {tol}")
            reliable &= res_mask

    elif problem == "lstsq":
        tol = _default_tol(x_fast.dtype) if opt_tol is None else opt_tol
        grad = residual @ A.conj()
        grad_norm = torch.linalg.norm(grad, dim=-1).to(real_dtype)
        denom = A_norm * (A_norm * x_norm + b_norm)
        rel_opt_error = grad_norm / denom.clamp_min(tiny)
        opt_mask = rel_opt_error <= tol
        if not opt_mask.all():
            print(f"[reliable lstsq] ||A^H r|| / (||A|| (||A||||x|| + ||b||)) > {tol}")
            reliable &= opt_mask
    else:
        raise ValueError(f"Unknown problem type: {problem}")

    if return_mask:
        return reliable
    return bool(reliable.all().item())


def spectral_filter_solve(
    A: torch.Tensor,
    b: torch.Tensor,
    kappa_eff: Optional[float] = None,
    gap_threshold: float = 10.0,
    prefer: Literal["auto", "hard", "soft"] = "auto",
    alpha: Optional[float] = None,
    rcond: Optional[float] = None,
    return_info: bool = False,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """SVD spectral-filter solve for A x ~= b, with row-wise b [bs, m]."""
    assert A.dim() == 2, f"Expected A to have shape [m,n], got {A.shape}"
    assert b.dim() == 2 and b.shape[1] == A.shape[0], f"Expected b to have shape [bs,m], got {b.shape}"

    original_dtype = b.dtype
    original_device = b.device
    is_complex = torch.is_complex(A) or torch.is_complex(b)
    work_dtype = torch.complex128 if is_complex else torch.float64
    real_work_dtype = torch.float64
    A_work = A.to(dtype=work_dtype, device=original_device)
    b_work = b.to(dtype=work_dtype, device=original_device)

    if kappa_eff is None: kappa_eff = 1e3
    if rcond is None: rcond = 1.0 / kappa_eff

    if str(original_device).startswith('cuda'):
        try:
            U, S, Vh = torch.linalg.svd(A_work, full_matrices=False, driver="gesvd")
        except:
            alpha_val = torch.tensor(0.01, dtype=work_dtype, device=original_device) * (A_work.abs().max() ** 2)
            alpha_val = torch.clamp(alpha_val, min=1e-3, max=5e-1)  # 防止太大或太小
            AtA = A_work.T @ A_work  # shape: [n, n]
            Atb = torch.einsum("mn,bm->bn", A_work, b_work) # shape: [bs, n]
            AtA_reg = AtA + alpha_val * torch.eye(A.shape[-1], dtype=work_dtype, device=original_device)  # shape: [n, n]
            AtA_reg_batch = AtA_reg.unsqueeze(0).expand(b_work.shape[0], -1, -1)  # [bs, n, n]
            Atb_batch = Atb.unsqueeze(-1)  # [bs, n, 1]
            sol = torch.linalg.lstsq(AtA_reg_batch, Atb_batch).solution  # [bs, n, 1]
            x = sol.squeeze(-1).to(dtype=original_dtype, device=original_device)  # [bs, n]
            return x, {}
            # print('gesvd failed, use cpu')
            # A_work = A_work.to('cpu')
            # U, S, Vh = torch.linalg.svd(A_work, full_matrices=False)
            # U, S, Vh = U.to(original_device), S.to(original_device), Vh.to(original_device)

            # sqrt_alpha = torch.sqrt(torch.tensor(alpha, dtype=A_work.dtype, device=A_work.device))
            # eye_reg = sqrt_alpha * torch.eye(n, dtype=A_work.dtype, device=A_work.device)
            # zero_reg = torch.zeros(n, dtype=b_work.dtype, device=b_work.device)
            # A_work = torch.cat([A_work, eye_reg], dim=0)  # shape: (m+n, n)
            # b_work = torch.cat([b_work, zero_reg], dim=0)  # shape: (m+n,)
    else:
       U, S, Vh = torch.linalg.svd(A_work, full_matrices=False)

    tiny = torch.finfo(S.dtype).tiny
    cond = S[0] / S[-1].clamp_min(tiny)

    sigma_cut = S[0] * rcond
    keep = S >= sigma_cut
    num_keep = max(int(keep.sum().item()), 1)

    gap_value = torch.tensor(float("nan"), dtype=S.dtype, device=S.device)
    has_gap = False
    if 0 < num_keep < S.numel():
        gap_value = S[num_keep - 1] / S[num_keep].clamp_min(tiny)
        has_gap = bool((gap_value >= gap_threshold).item())

    if prefer == "hard":
        method = "tsvd"
    elif prefer == "soft":
        method = "tikhonov"
    elif prefer == "auto":
        if bool((cond <= kappa_eff).item()):
            method = "full_svd"
        elif has_gap:
            method = "tsvd"
        else:
            method = "tikhonov"
    else:
        raise ValueError(f"Unknown prefer={prefer}")

    used_alpha = torch.tensor(float("nan"), dtype=S.dtype, device=S.device)
    if method == "full_svd":
        filt = 1.0 / S.clamp_min(tiny)
        effective_rank = S.numel()
    elif method == "tsvd":
        filt = torch.zeros_like(S)
        filt[:num_keep] = 1.0 / S[:num_keep].clamp_min(tiny)
        effective_rank = num_keep
    elif method == "tikhonov":
        used_alpha = sigma_cut.square() if alpha is None else torch.tensor(alpha, dtype=S.dtype, device=S.device)
        filt = S / (S.square() + used_alpha)
        effective_rank = S.numel()
    else:
        raise RuntimeError("Unexpected method.")

    coeff = b_work @ U.conj()
    x_work = (coeff * filt.to(work_dtype).unsqueeze(0)) @ Vh.conj()
    x = x_work.to(dtype=original_dtype, device=original_device)

    if return_info:
        info: Dict[str, torch.Tensor] = {
            "method": torch.tensor({"full_svd": 0, "tsvd": 1, "tikhonov": 2}[method], device=original_device),
            "sigma_max": S[0].detach().to(original_device),
            "sigma_min": S[-1].detach().to(original_device),
            "condition_number": cond.detach().to(original_device),
            "kappa_eff": torch.tensor(kappa_eff, dtype=real_work_dtype, device=original_device),
            "rcond": torch.tensor(rcond, dtype=real_work_dtype, device=original_device),
            "sigma_cut": sigma_cut.detach().to(original_device),
            "effective_rank": torch.tensor(effective_rank, device=original_device),
            "num_singular_values": torch.tensor(S.numel(), device=original_device),
            "has_gap": torch.tensor(has_gap, device=original_device),
            "gap_value": gap_value.detach().to(original_device),
            "alpha": used_alpha.detach().to(original_device),
        }
        return x, info
    return x, {}


def get_boundary_mask(xx, yy, num_kr, num_kc, num_rows, num_cols, PADDING=0, DEVICE='cpu'):
    top_row = max(xx-PADDING, 0)
    left_col = max(yy-PADDING, 0)
    bottom_row = max(num_kr-1-xx-PADDING, 0)
    right_col = max(num_kc-1-yy-PADDING, 0)
    bound = torch.zeros((num_rows, num_cols), dtype=torch.bool, device=DEVICE)
    bound[:top_row,:] = True
    bound[:,:left_col] = True
    if bottom_row>0:  bound[-bottom_row:,:] = True
    if right_col>0:  bound[:,-right_col:] = True
    return bound


def build_conv2d_matrix(_layer_params, input_shape):
    """Build the exact linear map used by torch conv2d for one sample."""
    _, c1, h1, w1 = input_shape
    weight = _layer_params.weight.detach()
    device = weight.device
    dtype = weight.dtype
    eye = torch.eye(c1 * h1 * w1, dtype=dtype, device=device)
    basis = eye.reshape(c1 * h1 * w1, c1, h1, w1)
    out = F.conv2d(
        basis,
        weight,
        bias=None,
        stride=_layer_params.stride,
        padding=_layer_params.padding,
        dilation=_layer_params.dilation,
        groups=_layer_params.groups,
    )
    return out.reshape(c1 * h1 * w1, -1).T.contiguous()


def _remove_bias(back_fea_star, _layer_params, _bias):
    if _bias and _layer_params.bias is not None:
        bias = _layer_params.bias.detach().reshape(1, -1, 1, 1)
        return back_fea_star - bias
    return back_fea_star


def convo_reverseCom_exact_shrink(front_fea, back_fea_star, _layer_params, _bias=True, jitter=0.0, kappa_eff=1e3, regu=True):
    """Nearest x to front_fea under the exact torch conv2d constraint."""
    bs = front_fea.shape[0]
    A = build_conv2d_matrix(_layer_params, front_fea.shape).to(torch.float64)  # c2*h2*w2, c1*h1*w1
    x0 = front_fea.detach().reshape(bs, -1).to(torch.float64)
    target = _remove_bias(back_fea_star.detach(), _layer_params, _bias).reshape(bs, -1).to(torch.float64)
    residual = target - (A @ x0.T).T

    try:
        gram = A @ A.T
        if jitter > 0:
            eye = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
            gram = gram + jitter * eye
        L = torch.linalg.cholesky(gram)
        lagrange = torch.cholesky_solve(residual.T, L)
        delta = (A.T @ lagrange).T
        # For shrink, a large correction amplification is expected for ill-conditioned A.
        # Do not reject an otherwise consistent correction just because norm_amp is high.
        if (not regu) or check_reliability(delta, A, residual, "solve", max_norm_amp=1e4):
            return (x0 + delta).reshape_as(front_fea).to(front_fea.dtype)
    except RuntimeError:
        print('[DEBUG] cholesky solve failed, use spectral filter solve')
    print('[DEBUG] exact shrink use spectral filter')
    # Prefer a less aggressive cutoff for exact_matrix shrink so the correction is not erased.
    delta = spectral_filter_solve(A, residual, prefer="auto", kappa_eff=float(kappa_eff))[0]
    return (x0 + delta).reshape_as(front_fea).to(front_fea.dtype)

def convo_reverseCom_exact_expand(front_fea, back_fea_star, _layer_params, _bias=True, kappa_eff=1e3, regu=True):
    """Least-squares x for the exact torch conv2d map."""
    bs = front_fea.shape[0]
    A = build_conv2d_matrix(_layer_params, front_fea.shape).to(torch.float64)
    target = _remove_bias(back_fea_star.detach(), _layer_params, _bias).reshape(bs, -1).to(torch.float64)
    try:
        x_new = torch.linalg.lstsq(A, target.T).solution.T
        if (not regu) or check_reliability(x_new, A, target, "lstsq", max_norm_amp=1e4):
            return x_new.reshape_as(front_fea).to(front_fea.dtype)
    except RuntimeError:
        print('[DEBUG] cholesky solve failed, use spectral filter solve')
        pass
    print('[DEBUG] exact expand use spectral filter')
    x_new = spectral_filter_solve(A, target, prefer="auto", kappa_eff=float(kappa_eff))[0]
    return x_new.reshape_as(front_fea).to(front_fea.dtype)


def conv2d_forward_linear(x, _layer_params):
    return F.conv2d(
        x,
        _layer_params.weight.detach().to(dtype=x.dtype, device=x.device),
        bias=None,
        stride=_layer_params.stride,
        padding=_layer_params.padding,
        dilation=_layer_params.dilation,
        groups=_layer_params.groups,
    )


def conv2d_adjoint_linear(y_grad, _layer_params, input_hw):
    padding = _layer_params.padding
    if isinstance(padding, int):
        padding = (padding, padding)
    stride = _layer_params.stride
    if isinstance(stride, int):
        stride = (stride, stride)
    dilation = _layer_params.dilation
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    kh, kw = _layer_params.weight.shape[-2:]
    h_in, w_in = input_hw
    h_out, w_out = y_grad.shape[-2:]
    output_padding = (
        h_in - ((h_out - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kh - 1) + 1),
        w_in - ((w_out - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kw - 1) + 1),
    )
    return F.conv_transpose2d(
        y_grad,
        _layer_params.weight.detach().to(dtype=y_grad.dtype, device=y_grad.device),
        bias=None,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=_layer_params.groups,
        dilation=dilation,
    )


def estimate_conv2d_operator_norm(_layer_params, input_shape, n_iter=10):
    """Power-iteration estimate of ||A|| for the bias-free conv operator."""
    device = _layer_params.weight.device
    x = torch.randn((1, *input_shape[1:]), dtype=torch.float64, device=device)
    x = x / _batch_flat_norm(x).reshape(-1, 1, 1, 1).clamp_min(1e-10)
    for _ in range(n_iter):
        y = conv2d_forward_linear(x, _layer_params)
        x = conv2d_adjoint_linear(y, _layer_params, input_shape[-2:])
        x = x / _batch_flat_norm(x).reshape(-1, 1, 1, 1).clamp_min(1e-10)
    y = conv2d_forward_linear(x, _layer_params)
    return (_batch_flat_norm(y) / _batch_flat_norm(x).clamp_min(1e-10)).amax()


def check_conv_reliability(
    x_fast: torch.Tensor,
    target: torch.Tensor,
    _layer_params,
    problem: Literal["solve", "lstsq"] = "solve",
    A_norm: Optional[torch.Tensor] = None,
    max_abs: Optional[float] = 1e2,
    max_norm_amp: Optional[float] = 1e4,
    residual_tol: Optional[float] = 0.1,
    opt_tol: Optional[float] = 0.1,
) -> bool:
    """ Operator-form reliability check for bias-free conv2d systems. """
    x_work = x_fast.detach().to(torch.float64)
    target_work = target.detach().to(torch.float64)
    if not torch.isfinite(x_work).all():
        print('[reliable conv] got non-finite values')
        return False
    if max_abs is not None and not (x_work.abs().amax(dim=tuple(range(1, x_work.ndim))) <= max_abs).all():
        print(f'[reliable conv] some value exceeds {max_abs}')
        return False

    if A_norm is None:
        A_norm = estimate_conv2d_operator_norm(_layer_params, x_work.shape)
    A_norm = torch.as_tensor(A_norm, dtype=torch.float64, device=x_work.device)
    tiny = torch.tensor(torch.finfo(torch.float64).eps, dtype=torch.float64, device=x_work.device)
    x_norm = _batch_flat_norm(x_work)
    target_norm = _batch_flat_norm(target_work)

    if max_norm_amp is not None:
        norm_amp = (A_norm * x_norm) / target_norm.clamp_min(tiny)
        if not (norm_amp <= max_norm_amp).all():
            print(f"[reliable conv] ||A|| ||x|| / ||b|| > {max_norm_amp}")
            return False

    residual = conv2d_forward_linear(x_work, _layer_params) - target_work

    if problem == "solve":
        tol = 1e-4 if residual_tol is None else residual_tol
        residual_norm = _batch_flat_norm(residual)
        rel = residual_norm / (A_norm * x_norm + target_norm).clamp_min(tiny)
        if not (rel <= tol).all():
            print(f"[reliable conv solve] relative residual > {tol}")
            return False
    elif problem == "lstsq":
        tol = 1e-4 if opt_tol is None else opt_tol
        grad = conv2d_adjoint_linear(residual, _layer_params, x_work.shape[-2:])
        grad_norm = _batch_flat_norm(grad)
        rel = grad_norm / (A_norm * (A_norm * x_norm + target_norm)).clamp_min(tiny)
        if not (rel <= tol).all():
            print(f"[reliable conv lstsq] normalized optimality residual > {tol}")
            return False
    else:
        raise ValueError(f"Unknown problem type: {problem}")
    return True


def _cg_damping_from_norm(A_norm, kappa_eff):
    A_norm = torch.as_tensor(A_norm, dtype=torch.float64)
    return float((A_norm / kappa_eff).square().clamp_min(torch.finfo(torch.float64).eps).item())


def _frequency_solve_with_fallback(
    A: torch.Tensor,
    B: torch.Tensor,
    problem: Literal["solve", "lstsq"],
    prefer: Literal["auto", "hard", "soft"] = "auto",
    max_norm_amp: Optional[float] = 1e4,
    kappa_eff: float = 1e2,
    force_regularized: bool = False,
):
    """Solve independent frequency systems with reliability-gated spectral filtering."""
    freq_shape = A.shape[:-2]
    m, n = A.shape[-2:]
    bs = B.shape[0]
    A_flat = A.reshape(-1, m, n)
    B_flat = B.reshape(bs, -1, m).transpose(0, 1).contiguous()
    X_chunks = []
    bad_count = 0

    for Ai, Bi in zip(A_flat, B_flat):
        Xi = None
        reliable = False
        if not force_regularized:
            try:
                if problem == "solve" and m == n:
                    Xi = torch.linalg.solve(Ai, Bi.mT).mT
                    reliable = check_reliability(Xi, Ai, Bi, "solve", max_abs=1e2, max_norm_amp=max_norm_amp)
                else:
                    Xi = torch.linalg.lstsq(Ai, Bi.mT, driver='gels').solution.mT
                    reliable = check_reliability(Xi, Ai, Bi, "lstsq", max_abs=1e2, max_norm_amp=max_norm_amp)
            except RuntimeError:
                reliable = False

        if force_regularized or not reliable:
            bad_count += 1
            Xi = spectral_filter_solve(Ai, Bi, prefer=prefer, kappa_eff=kappa_eff)[0]
        X_chunks.append(Xi)

    if bad_count:
        action = "regularized" if force_regularized else "spectral fallback used for"
        print(f"[DEBUG fft] {action} {bad_count}/{A_flat.shape[0]} frequency systems")
    X_flat = torch.stack(X_chunks, dim=1)
    return X_flat.reshape(bs, *freq_shape, n)


def conjugate_gradient_batched(apply_A, b, max_iter=200, tol=1e-10, damping=0.0):
    """Batched CG for symmetric positive semidefinite systems."""
    x = torch.zeros_like(b)
    r = b - (apply_A(x) + damping * x)
    p = r.clone()
    reduce_dims = tuple(range(1, b.ndim))
    rs_old = torch.sum(r * r, dim=reduce_dims)
    b_norm = torch.sqrt(torch.clamp(torch.sum(b * b, dim=reduce_dims), min=1e-10))
    for _ in range(max_iter):
        Ap = apply_A(p) + damping * p
        denom = torch.sum(p * Ap, dim=reduce_dims).clamp_min(1e-10)
        alpha = (rs_old / denom).reshape(-1, *([1] * (b.ndim - 1)))
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.sum(r * r, dim=reduce_dims)
        if torch.max(torch.sqrt(torch.clamp(rs_new, min=0.0)) / b_norm.clamp_min(1e-10)) < tol:
            break
        beta = (rs_new / rs_old.clamp_min(1e-30)).reshape(-1, *([1] * (b.ndim - 1)))
        p = r + beta * p
        rs_old = rs_new
    return x


def convo_reverseCom_cg_shrink(front_fea, back_fea_star, _layer_params, _bias=True, max_iter=300, tol=1e-10, regu=True):
    """Nearest x to front_fea using CG on A A^T, with reliability-gated damping."""
    x0 = front_fea.detach().to(torch.float64)
    target = _remove_bias(back_fea_star.detach(), _layer_params, _bias).to(torch.float64)
    residual = target - conv2d_forward_linear(x0, _layer_params)
    A_norm = estimate_conv2d_operator_norm(_layer_params, x0.shape)

    def apply_AAT(alpha):
        return conv2d_forward_linear(conv2d_adjoint_linear(alpha, _layer_params, x0.shape[-2:]), _layer_params)

    alpha = conjugate_gradient_batched(apply_AAT, residual, max_iter=max_iter, tol=tol)
    delta = conv2d_adjoint_linear(alpha, _layer_params, x0.shape[-2:])
    # Shrink solves A delta = residual. High norm_amp is diagnostic, not by itself a reason
    # to discard a correction that gives good compute consistency.
    if (not regu) or check_conv_reliability(delta, residual, _layer_params, "solve", A_norm=A_norm, max_norm_amp=1e4):
        return (x0 + delta).to(front_fea.dtype)

    print("[DEBUG cg shrink] retry with stronger Tikhonov damping")
    damping = _cg_damping_from_norm(A_norm, kappa_eff=1e3)
    alpha = conjugate_gradient_batched(apply_AAT, residual, max_iter=max_iter, tol=tol, damping=damping)
    delta = conv2d_adjoint_linear(alpha, _layer_params, x0.shape[-2:])
    return (x0 + delta).to(front_fea.dtype)


def convo_reverseCom_cg_expand(front_fea, back_fea_star, _layer_params, _bias=True, max_iter=300, tol=1e-10, regu=True):
    """Least-squares x using CG on A^T A, with reliability-gated damping."""
    target = _remove_bias(back_fea_star.detach(), _layer_params, _bias).to(torch.float64)
    input_hw = front_fea.shape[-2:]
    input_shape = front_fea.shape
    rhs = conv2d_adjoint_linear(target, _layer_params, input_hw)
    A_norm = estimate_conv2d_operator_norm(_layer_params, input_shape)

    def apply_ATA(x):
        return conv2d_adjoint_linear(conv2d_forward_linear(x, _layer_params), _layer_params, input_hw)

    x_new = conjugate_gradient_batched(apply_ATA, rhs, max_iter=max_iter, tol=tol)
    if (not regu) or check_conv_reliability(x_new, target, _layer_params, "lstsq", A_norm=A_norm, max_norm_amp=1e4):
        return x_new.to(front_fea.dtype)

    print("[DEBUG cg expand] retry with stronger Tikhonov damping")
    damping = _cg_damping_from_norm(A_norm, kappa_eff=1e3)
    x_new = conjugate_gradient_batched(apply_ATA, rhs, max_iter=max_iter, tol=tol, damping=damping)
    return x_new.to(front_fea.dtype)


def _fft_kappa_eff(kappa_eff):
    """Frequency systems are tiny, so use a stricter spectral cutoff by default."""
    return min(float(kappa_eff), 100)


def _fft_result_reliable(result, front_fea, back_fea_star, _layer_params, _bias, mode):
    target = _remove_bias(back_fea_star.detach(), _layer_params, _bias).to(torch.float64)
    problem = "solve" if mode == "shrink" else "lstsq"
    return check_conv_reliability(
        result.to(torch.float64),
        target,
        _layer_params,
        problem,
        max_abs=1e2,
        max_norm_amp=1e4,
        residual_tol=1e-1,
        opt_tol=1e-1,
    )


def convo_reverseCom_shrink(front_fea, back_fea_star, _layer_params, _bias=True, kappa_eff=1e3, regu=True):
    """FFT old-bound shrink path with per-frequency spectral fallback."""
    DEVICE = front_fea.device
    PADDING = _layer_params.padding[0]
    bs, c1, h1, w1 = front_fea.shape
    c2, _, hw, ww = _layer_params.weight.shape
    P_H = torch.eye(h1, dtype=torch.float64, device=DEVICE).unsqueeze(0).unsqueeze(0)
    P_H = torch.roll(P_H, shifts=PADDING, dims=-1)
    fft_P_H = torch.fft.fft2(P_H, dim=[-2, -1])
    P_W = torch.eye(w1, dtype=torch.float64, device=DEVICE).unsqueeze(0).unsqueeze(0)
    P_W = torch.roll(P_W, shifts=-PADDING, dims=-1)
    fft_P_W = torch.fft.fft2(P_W, dim=[-2, -1])

    pad = nn.ZeroPad2d((w1 - back_fea_star.shape[-1], 0, h1 - back_fea_star.shape[-2], 0))
    if _bias:
        bias = _layer_params.bias.detach().reshape(1, -1, 1, 1).expand(bs, -1, back_fea_star.shape[-2], back_fea_star.shape[-1])
        fft_back = torch.fft.fft2(pad(back_fea_star - bias), dim=[-2, -1])
    else:
        fft_back = torch.fft.fft2(pad(back_fea_star), dim=[-2, -1])
    fft_back = (fft_P_W.mH @ fft_back.to(torch.complex128).mH @ fft_P_H.mH).transpose(-2, -1) / h1 / w1
    fft_back = fft_back.permute(0, 2, 3, 1)
    fft_front_old = torch.fft.fft2(front_fea.to(torch.float64), dim=[-2, -1]).permute(0, 2, 3, 1)
    reverse_weight = torch.flip(_layer_params.weight.detach().to(torch.float64), [-2, -1])
    fft_rev_weight = torch.fft.fft2(reverse_weight, s=(h1, w1), dim=[-2, -1]).permute(2, 3, 0, 1)

    A = torch.zeros(h1, w1, c1 + c2, c1 + c2, dtype=torch.complex128, device=DEVICE)
    A[:, :, torch.arange(c1, device=DEVICE), torch.arange(c1, device=DEVICE)] = 1.0
    A[:, :, c1:, :c1] = fft_rev_weight
    A[:, :, :c1, c1:] = fft_rev_weight.transpose(-1, -2)
    
    G = torch.zeros(bs, c2, c1, h1, w1, dtype=torch.complex128, device=DEVICE)
    if PADDING > 0:
        rows = torch.arange(h1, device=DEVICE).reshape(-1, 1)
        cols = torch.arange(w1, device=DEVICE).reshape(1, -1)
        Eh = torch.exp(torch.complex(torch.tensor(0.0, device=DEVICE), torch.tensor(-2 * torch.pi / h1, device=DEVICE)))
        Ew = torch.exp(torch.complex(torch.tensor(0.0, device=DEVICE), torch.tensor(-2 * torch.pi / w1, device=DEVICE)))
        for i in range(hw):
            for j in range(ww):
                bound = get_boundary_mask(hw - i - 1, ww - j - 1, hw, ww, h1, w1, PADDING, DEVICE)
                G += reverse_weight[:, :, i, j].unsqueeze(0).unsqueeze(-1).unsqueeze(-1) * \
                    (Eh ** (i * rows) * Ew ** (j * cols)).unsqueeze(0).unsqueeze(0).unsqueeze(0) * \
                    torch.fft.fft2(front_fea.to(torch.float64) * bound, dim=[-2, -1]).unsqueeze(1)
    Gm_sum = torch.sum(G, dim=2).permute(0, 2, 3, 1)
    B = torch.cat([fft_front_old, Gm_sum + fft_back], dim=-1).to(torch.complex128)
    # fft_res = _frequency_solve_with_fallback(A, B, "solve", prefer="auto", kappa_eff=_fft_kappa_eff(kappa_eff))[:, :, :, :c1]
    fft_res = torch.linalg.solve(A.unsqueeze(0), B.unsqueeze(-1))[:,:,:,:c1,:].squeeze(-1)  # bs, u, v, c1,1
    fft_res = fft_res.permute(0, 3, 1, 2)
    result = torch.fft.ifft2(fft_res, dim=[-2, -1]).real
    if regu and not _fft_result_reliable(result, front_fea, back_fea_star, _layer_params, _bias, "shrink"):
        fft_res = _frequency_solve_with_fallback(
            A, B, "solve", prefer="soft", kappa_eff=_fft_kappa_eff(kappa_eff), force_regularized=True
        )[:, :, :, :c1]
        result = torch.fft.ifft2(fft_res.permute(0, 3, 1, 2), dim=[-2, -1]).real
    return result.to(front_fea.dtype)


def convo_reverseCom_expand(front_fea, back_fea_star, _layer_params, _bias=True, kappa_eff=1e3, regu=True):
    """FFT old-bound expand path with per-frequency spectral fallback."""
    DEVICE = back_fea_star.device
    PADDING = _layer_params.padding[0]
    bs, c1, h1, w1 = front_fea.shape
    c2, _, hw, ww = _layer_params.weight.shape
    P_H = torch.eye(h1, dtype=torch.float64, device=DEVICE).unsqueeze(0).unsqueeze(0)
    P_H = torch.roll(P_H, shifts=PADDING, dims=-1)
    fft_P_H = torch.fft.fft2(P_H, dim=[-2, -1])
    P_W = torch.eye(w1, dtype=torch.float64, device=DEVICE).unsqueeze(0).unsqueeze(0)
    P_W = torch.roll(P_W, shifts=-PADDING, dims=-1)
    fft_P_W = torch.fft.fft2(P_W, dim=[-2, -1])

    pad = nn.ZeroPad2d((w1 - back_fea_star.shape[-1], 0, h1 - back_fea_star.shape[-2], 0))
    if _bias:
        bias = _layer_params.bias.detach().reshape(1, -1, 1, 1).expand(bs, -1, back_fea_star.shape[-2], back_fea_star.shape[-1])
        fft_back = torch.fft.fft2(pad(back_fea_star - bias), dim=[-2, -1])
    else:
        fft_back = torch.fft.fft2(pad(back_fea_star), dim=[-2, -1])
    fft_back = (fft_P_W.mH @ fft_back.to(torch.complex128).mH @ fft_P_H.mH).transpose(-2, -1) / h1 / w1
    fft_back = fft_back.permute(0, 2, 3, 1)
    reverse_weight = torch.flip(_layer_params.weight.detach().to(torch.float64), [-2, -1])
    fft_rev_weight = torch.fft.fft2(reverse_weight, s=(h1, w1), dim=[-2, -1]).permute(2, 3, 0, 1)

    rows = torch.arange(h1, device=DEVICE).reshape(-1, 1)
    cols = torch.arange(w1, device=DEVICE).reshape(1, -1)
    Eh = torch.exp(torch.complex(torch.tensor(0.0, device=DEVICE), torch.tensor(-2 * torch.pi / h1, device=DEVICE)))
    Ew = torch.exp(torch.complex(torch.tensor(0.0, device=DEVICE), torch.tensor(-2 * torch.pi / w1, device=DEVICE)))
    G = torch.zeros(bs, c2, c1, h1, w1, dtype=torch.complex128, device=DEVICE)
    for i in range(hw):
        for j in range(ww):
            bound = get_boundary_mask(hw - i - 1, ww - j - 1, hw, ww, h1, w1, PADDING, DEVICE)
            G += reverse_weight[:, :, i, j].unsqueeze(0).unsqueeze(-1).unsqueeze(-1) * \
                (Eh ** (i * rows) * Ew ** (j * cols)).unsqueeze(0).unsqueeze(0).unsqueeze(0) * \
                torch.fft.fft2(front_fea.to(torch.float64) * bound, dim=[-2, -1]).unsqueeze(1)
    B = fft_back + torch.sum(G, dim=2).permute(0, 2, 3, 1)
    # fft_res = _frequency_solve_with_fallback(fft_rev_weight, B, "lstsq", prefer="auto", kappa_eff=_fft_kappa_eff(kappa_eff))
    fft_res = torch.linalg.lstsq(fft_rev_weight.unsqueeze(0).expand(bs, -1, -1,-1,-1),\
                B.unsqueeze(-1), driver='gels').solution.squeeze(-1)  # bs, u, v, c1
    fft_res = fft_res.permute(0, 3, 1, 2)
    result = torch.fft.ifft2(fft_res, dim=[-2, -1]).real
    if regu and not _fft_result_reliable(result, front_fea, back_fea_star, _layer_params, _bias, "expand"):
        fft_res = _frequency_solve_with_fallback(
            fft_rev_weight, B, "lstsq", prefer="soft", kappa_eff=_fft_kappa_eff(kappa_eff), force_regularized=True
        )
        result = torch.fft.ifft2(fft_res.permute(0, 3, 1, 2), dim=[-2, -1]).real
    return result.to(front_fea.dtype)


def convo_reverseCom_clip_shrink(front_fea, back_fea_star, _layer_params, _bias=True, kappa_eff=1e3, regu=True):
    """Padded FFT shrink path with per-frequency spectral fallback."""
    DEVICE = front_fea.device
    PADDING = _layer_params.padding[0]
    original_dtype = front_fea.dtype
    pad_x = nn.ZeroPad2d((PADDING, PADDING, PADDING, PADDING))
    front_padded = pad_x(front_fea.to(torch.float64))
    bs, c1, h1, w1 = front_padded.shape
    c2, _, hw, ww = _layer_params.weight.shape

    pad = nn.ZeroPad2d((w1 - back_fea_star.shape[-1], 0, h1 - back_fea_star.shape[-2], 0))
    if _bias:
        bias = _layer_params.bias.detach().reshape(1, -1, 1, 1).expand(bs, -1, back_fea_star.shape[-2], back_fea_star.shape[-1])
        fft_back = torch.fft.fft2(pad(back_fea_star - bias), dim=[-2, -1])
    else:
        fft_back = torch.fft.fft2(pad(back_fea_star), dim=[-2, -1])
    fft_back = fft_back.to(torch.complex128).permute(0, 2, 3, 1)
    fft_front_old = torch.fft.fft2(front_padded, dim=[-2, -1]).permute(0, 2, 3, 1)
    reverse_weight = torch.flip(_layer_params.weight.detach().to(torch.float64), [-2, -1])
    fft_rev_weight = torch.fft.fft2(reverse_weight, s=(h1, w1), dim=[-2, -1]).permute(2, 3, 0, 1)

    A = torch.zeros(h1, w1, c1 + c2, c1 + c2, dtype=torch.complex128, device=DEVICE)
    A[:, :, torch.arange(c1, device=DEVICE), torch.arange(c1, device=DEVICE)] = 1.0
    A[:, :, c1:, :c1] = fft_rev_weight
    A[:, :, :c1, c1:] = fft_rev_weight.transpose(-1, -2)
    B = torch.cat([fft_front_old, fft_back], dim=-1).to(torch.complex128)
    # fft_res = _frequency_solve_with_fallback(A, B, "solve", prefer="soft", kappa_eff=_fft_kappa_eff(kappa_eff))[:, :, :, :c1]
    fft_res = torch.linalg.solve(A.unsqueeze(0), B.unsqueeze(-1))[:,:,:,:c1,:].squeeze(-1)  # bs, u, v, c1,1
    fft_res = fft_res.permute(0, 3, 1, 2)
    recons_front_fea = torch.fft.ifft2(fft_res, dim=[-2, -1]).real
    if PADDING > 0:
        result = recons_front_fea[:, :, PADDING:-PADDING, PADDING:-PADDING]
    else:
        result = recons_front_fea
    if regu and not _fft_result_reliable(result, front_fea, back_fea_star, _layer_params, _bias, "shrink"):
        fft_res = _frequency_solve_with_fallback(
            A, B, "solve", prefer="soft", kappa_eff=_fft_kappa_eff(kappa_eff), force_regularized=True
        )[:, :, :, :c1]
        recons_front_fea = torch.fft.ifft2(fft_res.permute(0, 3, 1, 2), dim=[-2, -1]).real
        result = recons_front_fea[:, :, PADDING:-PADDING, PADDING:-PADDING] if PADDING > 0 else recons_front_fea
    return result.to(front_fea.dtype)


def convo_reverseCom_clip_expand(front_fea, back_fea_star, _layer_params, _bias=True, kappa_eff=1e3, regu=True):
    """Padded FFT expand path with per-frequency spectral fallback."""
    PADDING = _layer_params.padding[0]
    original_dtype = front_fea.dtype
    pad_x = nn.ZeroPad2d((PADDING, PADDING, PADDING, PADDING))
    front_padded = pad_x(front_fea.to(torch.float64))
    bs, c1, h1, w1 = front_padded.shape

    pad = nn.ZeroPad2d((w1 - back_fea_star.shape[-1], 0, h1 - back_fea_star.shape[-2], 0))
    if _bias:
        bias = _layer_params.bias.detach().reshape(1, -1, 1, 1).expand(bs, -1, back_fea_star.shape[-2], back_fea_star.shape[-1])
        fft_back = torch.fft.fft2(pad(back_fea_star - bias), dim=[-2, -1])
    else:
        fft_back = torch.fft.fft2(pad(back_fea_star), dim=[-2, -1])
    B = fft_back.to(torch.complex128).permute(0, 2, 3, 1)  # bs, u, v, c2
    reverse_weight = torch.flip(_layer_params.weight.detach().to(torch.float64), [-2, -1])
    fft_rev_weight = torch.fft.fft2(reverse_weight, s=(h1, w1), dim=[-2, -1]).permute(2, 3, 0, 1)  # u, v, c2, c1
    fft_res = torch.linalg.lstsq(fft_rev_weight.unsqueeze(0).expand(bs, -1, -1,-1,-1),\
                B.unsqueeze(-1), driver='gels').solution.squeeze(-1)  # bs, u, v, c1
    fft_res = fft_res.permute(0, 3, 1, 2)
    recons_front_fea = torch.fft.ifft2(fft_res, dim=[-2, -1]).real
    if PADDING == 0:
        result = recons_front_fea
    else:
        result = recons_front_fea[:, :, PADDING:-PADDING, PADDING:-PADDING]
    if regu and not _fft_result_reliable(result, front_fea, back_fea_star, _layer_params, _bias, "expand"):
        fft_res = _frequency_solve_with_fallback(
            fft_rev_weight, B, "lstsq", prefer="soft", kappa_eff=_fft_kappa_eff(kappa_eff), force_regularized=True
        )
        recons_front_fea = torch.fft.ifft2(fft_res.permute(0, 3, 1, 2), dim=[-2, -1]).real
        result = recons_front_fea if PADDING == 0 else recons_front_fea[:, :, PADDING:-PADDING, PADDING:-PADDING]
    return result.to(front_fea.dtype)


def convo_reverseCom(front_fea, back_fea_star, _layer_params, _bias=True, method='cg', kappa_eff=1e3, regu=True):
    """must be: front_fea.shape[-2:] > back_fea_star.shape[-2:]"""
    assert _layer_params.groups == 1, "FFT path currently supports groups=1 only"
    assert _layer_params.stride == (1, 1), "FFT path currently supports stride=1 only"
    assert _layer_params.dilation == (1, 1), "FFT path currently supports dilation=1 only"
    if method == 'exact_matrix':
        if torch.prod(torch.tensor(front_fea.shape[1:])).item() >= torch.prod(torch.tensor(back_fea_star.shape[1:])).item():
            return convo_reverseCom_exact_shrink(front_fea, back_fea_star, _layer_params, _bias, kappa_eff=kappa_eff, regu=regu)
        return convo_reverseCom_exact_expand(front_fea, back_fea_star, _layer_params, _bias, kappa_eff=kappa_eff, regu=regu)
    if method == 'cg':
        if torch.prod(torch.tensor(front_fea.shape[1:])).item() >= torch.prod(torch.tensor(back_fea_star.shape[1:])).item():
            return convo_reverseCom_cg_shrink(front_fea, back_fea_star, _layer_params, _bias, max_iter=100, tol=1e-9, regu=regu)
        return convo_reverseCom_cg_expand(front_fea, back_fea_star, _layer_params, _bias, max_iter=100, tol=1e-9, regu=regu)
    if method == "fft_oldbound":
        if front_fea.shape[1] >= back_fea_star.shape[1]:
            return convo_reverseCom_shrink(front_fea, back_fea_star, _layer_params, _bias, kappa_eff=kappa_eff, regu=regu)
        return convo_reverseCom_expand(front_fea, back_fea_star, _layer_params, _bias, kappa_eff=kappa_eff, regu=regu)
    if method == "fft_pad":
        if front_fea.shape[1] >= back_fea_star.shape[1]:
            return convo_reverseCom_clip_shrink(front_fea, back_fea_star, _layer_params, _bias, kappa_eff=kappa_eff, regu=regu)
        return convo_reverseCom_clip_expand(front_fea, back_fea_star, _layer_params, _bias, kappa_eff=kappa_eff, regu=regu)
    raise ValueError("method must be one of: 'cg', 'fft_oldbound', 'exact_matrix', 'fft_pad'")
