import math
from typing import Dict, Literal, Optional, Tuple

import torch


def remove_bias(back_fea_star, _layer_params):
    if _layer_params.bias is None:
        return back_fea_star
    bias = _layer_params.bias.detach().reshape(1, -1, *([1] * (back_fea_star.ndim - 2)))
    return back_fea_star - bias


def check_reliability(
    x_fast: torch.Tensor,
    A: torch.Tensor,
    b: torch.Tensor,
    problem: Literal["solve", "lstsq"] = "solve",
    x_ref: Optional[torch.Tensor] = None,
    max_abs: Optional[float] = 1e3,
    amplification_tol: float = 1e2,
) -> torch.Tensor:
    assert x_fast.ndim == A.ndim == b.ndim == 2
    assert x_fast.shape == (b.shape[0], A.shape[1]) and b.shape[1] == A.shape[0]

    real_dtype = x_fast.real.dtype
    tiny = torch.finfo(real_dtype).eps
    reliable = torch.isfinite(x_fast).all(dim=-1)
    if max_abs is not None:
        reliable &= x_fast.abs().amax(dim=-1) <= max_abs
    if x_ref is not None and amplification_tol is not None:
        delta_norm = torch.linalg.vector_norm(x_fast - x_ref, dim=-1)
        ref_norm = torch.linalg.vector_norm(x_ref, dim=-1)
        ref_floor = 1e-2 * math.sqrt(x_ref.shape[-1])
        reliable &= (ref_norm <= ref_floor) | (delta_norm <= amplification_tol * ref_norm)

    residual = x_fast @ A.mT - b
    a_norm = torch.linalg.matrix_norm(A, ord="fro")
    x_norm = torch.linalg.vector_norm(x_fast, dim=-1)
    b_norm = torch.linalg.vector_norm(b, dim=-1)
    if problem == "solve":
        error = torch.linalg.vector_norm(residual, dim=-1)
        tol = 1e-4 if real_dtype == torch.float64 else 1e-3
        return reliable & (error / (a_norm * x_norm + b_norm).clamp_min(tiny) <= tol)
    if problem == "lstsq":
        grad = residual @ A.conj()
        error = torch.linalg.vector_norm(grad, dim=-1)
        tol = 1e-3 if real_dtype == torch.float64 else 1e-2
        return reliable & (error / (a_norm * (a_norm * x_norm + b_norm)).clamp_min(tiny) <= tol)
    raise ValueError(f"Unknown problem type: {problem}")


def check_fft_reliablity(
    x_fast: torch.Tensor,
    A: torch.Tensor,
    b: torch.Tensor,
    problem: Literal["solve", "lstsq"] = "solve",
    x_ref: Optional[torch.Tensor] = None,
    max_abs: Optional[float] = 1e3,
    amplification_tol: float = 1e2,
    a_norm: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Frequency-wise reliability. A may be [F,m,n] or [F,B,m,n]."""
    assert x_fast.ndim == b.ndim == 3 and A.ndim in (3, 4)
    if A.ndim == 3:
        assert A.shape[0] == x_fast.shape[0] == b.shape[0]
        assert x_fast.shape[-1] == A.shape[-1] and b.shape[-1] == A.shape[-2]
    else:
        assert A.shape[:-2] == x_fast.shape[:-1] == b.shape[:-1]
        assert x_fast.shape[-1] == A.shape[-1] and b.shape[-1] == A.shape[-2]

    real_dtype = x_fast.real.dtype
    tiny = torch.finfo(real_dtype).eps
    reliable = torch.isfinite(x_fast).all(dim=-1)
    if max_abs is not None:
        reliable &= x_fast.abs().amax(dim=-1) <= max_abs
    if x_ref is not None and amplification_tol is not None:
        delta_norm = torch.linalg.vector_norm(x_fast - x_ref, dim=-1)
        ref_norm = torch.linalg.vector_norm(x_ref, dim=-1)
        ref_floor = 1e-2 * math.sqrt(x_ref.shape[-1])
        reliable &= (ref_norm <= ref_floor) | (delta_norm <= amplification_tol * ref_norm)

    if A.ndim == 3:
        residual = torch.einsum("fmn,fbn->fbm", A, x_fast) - b
        if a_norm is None:
            a_norm = torch.linalg.matrix_norm(A, ord="fro")
        a_norm = a_norm.reshape(-1, 1)
    else:
        residual = (A @ x_fast.unsqueeze(-1)).squeeze(-1) - b
        if a_norm is None:
            a_norm = torch.linalg.matrix_norm(A, ord="fro")

    x_norm = torch.linalg.vector_norm(x_fast, dim=-1)
    b_norm = torch.linalg.vector_norm(b, dim=-1)
    if problem == "solve":
        error = torch.linalg.vector_norm(residual, dim=-1)
        tol = 1e-4 if real_dtype == torch.float64 else 1e-3
        return reliable & (error / (a_norm * x_norm + b_norm).clamp_min(tiny) <= tol)
    if problem == "lstsq":
        if A.ndim == 3:
            grad = torch.einsum("fmn,fbm->fbn", A.conj(), residual)
        else:
            grad = (A.conj().transpose(-2, -1) @ residual.unsqueeze(-1)).squeeze(-1)
        error = torch.linalg.vector_norm(grad, dim=-1)
        tol = 1e-3 if real_dtype == torch.float64 else 1e-2
        return reliable & (error / (a_norm * (a_norm * x_norm + b_norm)).clamp_min(tiny) <= tol)
    raise ValueError(f"Unknown problem type: {problem}")


def _svd(A):
    args = {"driver": "gesvd"} if A.is_cuda else {}
    try:
        return torch.linalg.svd(A, full_matrices=False, **args)
    except (torch.linalg.LinAlgError, RuntimeError):
        if not A.is_cuda:
            raise
        U, S, Vh = torch.linalg.svd(A.detach().cpu(), full_matrices=False)
        return U.to(A.device), S.to(A.device), Vh.to(A.device)


def mdp_tikhonov_solve(
    A: torch.Tensor,
    b: torch.Tensor,
    x0: torch.Tensor,
    kappa_eff: float = 1e3,
    alpha=None,
    alpha_scale: float = 1.0,
    max_abs: Optional[float] = 1e3,
    return_info: bool = False,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Shared centered Tikhonov solve with per-sample alpha."""
    if A.ndim != 2 or b.ndim != 2 or x0.ndim != 2:
        raise ValueError("Expected A[m,n], b[B,m], and x0[B,n].")
    m, n = A.shape
    if b.shape[1:] != (m,) or x0.shape != (b.shape[0], n):
        raise ValueError(f"Incompatible shapes: A={tuple(A.shape)}, b={tuple(b.shape)}, x0={tuple(x0.shape)}")
    if kappa_eff <= 0 or alpha_scale <= 0:
        raise ValueError("Require kappa_eff/alpha_scale > 0.")

    U, S, Vh = _svd(A)
    eps = torch.finfo(S.dtype).eps
    sigma_max = S[0] if S.numel() else S.new_zeros(())
    if alpha is None:
        base = torch.maximum((sigma_max / kappa_eff).square(), eps * sigma_max.square().clamp_min(1))
        used_alpha = base.expand(b.shape[0]).clone()
    else:
        used_alpha = torch.as_tensor(alpha, dtype=S.dtype, device=A.device)
        if used_alpha.ndim == 0:
            used_alpha = used_alpha.expand(b.shape[0]).clone()
        elif used_alpha.shape != (b.shape[0],):
            raise ValueError("alpha must be a scalar or have shape [B].")
        if not torch.isfinite(used_alpha).all() or (used_alpha <= 0).any():
            raise ValueError("alpha must contain finite positive values.")
    used_alpha *= alpha_scale

    residual = b - x0 @ A.mT
    if max_abs is not None:
        margin = max_abs - x0.abs().amax(dim=-1).real
        bound = (torch.linalg.vector_norm(residual, dim=-1).real / (2 * margin.clamp_min(eps))).square()
        used_alpha = torch.where(margin > 0, torch.maximum(used_alpha, bound), used_alpha)

    coeff = residual @ U.conj()
    delta = (coeff * (S / (S.square() + used_alpha.unsqueeze(-1)))) @ Vh.conj()
    x = x0 + delta
    if not return_info:
        return x, {}

    sigma_min = S[-1] if S.numel() else S.new_zeros(())
    tiny = torch.finfo(S.dtype).tiny
    final_residual = x @ A.mT - b
    residual_norm = torch.linalg.vector_norm(final_residual, dim=-1).real
    return x, {
        "sigma_max": sigma_max.detach(),
        "sigma_min": sigma_min.detach(),
        "condition_number": (sigma_max / sigma_min.clamp_min(tiny)).detach(),
        "residual_norm": residual_norm.detach(),
        "relative_residual": (residual_norm / torch.linalg.vector_norm(b, dim=-1).real.clamp_min(tiny)).detach(),
        "delta_norm": torch.linalg.vector_norm(delta, dim=-1).real.detach(),
    }


def mdp_tikhonov_fft_solve(
    A: torch.Tensor,
    b: torch.Tensor,
    x0: torch.Tensor,
    kappa_eff: float = 1e3,
    alpha=None,
    alpha_scale: float = 1.0,
    max_abs: Optional[float] = 1e3,
    return_info: bool = False,
    system_index: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Batched FFT Tikhonov solve; system_index lets many RHS share one SVD."""
    m, n = A.shape[-2:]
    out_shape = x0.shape
    b_flat, x0_flat = b.reshape(-1, m), x0.reshape(-1, n)

    if system_index is None:
        if A.shape[:-2] != b.shape[:-1] or A.shape[:-2] != x0.shape[:-1]:
            raise ValueError(f"Incompatible shapes: A={tuple(A.shape)}, b={tuple(b.shape)}, x0={tuple(x0.shape)}")
        A_sys = A.reshape(-1, m, n)
        U, S, Vh = _svd(A_sys)
        A_pair = A_sys
    else:
        A_sys = A.reshape(-1, m, n)
        idx = system_index.reshape(-1).long()
        if idx.numel() != b_flat.shape[0] or idx.min() < 0 or idx.max() >= A_sys.shape[0]:
            raise ValueError("Invalid system_index.")
        U0, S0, Vh0 = _svd(A_sys)
        U, S, Vh = U0[idx], S0[idx], Vh0[idx]
        A_pair = A_sys[idx]

    eps = torch.finfo(S.dtype).eps
    sigma_max = S[..., 0]
    if alpha is None:
        used_alpha = torch.maximum((sigma_max / kappa_eff).square(), eps * sigma_max.square().clamp_min(1))
    else:
        used_alpha = torch.broadcast_to(torch.as_tensor(alpha, dtype=S.dtype, device=A.device), sigma_max.shape).clone()
    used_alpha *= alpha_scale

    residual = b_flat - (A_pair @ x0_flat.unsqueeze(-1)).squeeze(-1)
    if max_abs is not None:
        margin = max_abs - x0_flat.abs().amax(dim=-1).real
        bound = (torch.linalg.vector_norm(residual, dim=-1).real / (2 * margin.clamp_min(eps))).square()
        used_alpha = torch.where(margin > 0, torch.maximum(used_alpha, bound), used_alpha)

    coeff = (residual.unsqueeze(-2) @ U.conj()).squeeze(-2)
    delta = ((coeff * (S / (S.square() + used_alpha.unsqueeze(-1)))).unsqueeze(-2) @ Vh.conj()).squeeze(-2)
    x = (x0_flat + delta).reshape(out_shape)
    if not return_info:
        return x, {}

    sigma_min = S[..., -1]
    tiny = torch.finfo(S.dtype).tiny
    stat_shape = out_shape[:-1]
    return x, {
        "sigma_max": sigma_max.reshape(stat_shape).detach(),
        "sigma_min": sigma_min.reshape(stat_shape).detach(),
        "condition_number": (sigma_max / sigma_min.clamp_min(tiny)).reshape(stat_shape).detach(),
    }


if __name__ == "__main__":
    torch.manual_seed(0)
    A = torch.eye(2)
    b = torch.tensor([[1.0, -2.0], [1.0, -2.0]])
    mask = check_reliability(torch.tensor([[1.0, -2.0], [0.0, 0.0]]), A, b)
    torch.testing.assert_close(mask, torch.tensor([True, False]))

    fft_A = torch.eye(2, dtype=torch.complex64).reshape(1, 2, 2).expand(6, -1, -1)
    fft_b = torch.ones(6, 2, dtype=torch.complex64)
    fft_x, _ = mdp_tikhonov_fft_solve(fft_A, fft_b, torch.zeros_like(fft_b))
    assert torch.isfinite(fft_x).all()
    print("[PASS] utils")
