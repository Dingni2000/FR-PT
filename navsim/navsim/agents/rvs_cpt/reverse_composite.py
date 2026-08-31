import math
import torch
import torch.nn as nn


def _tikhonov_alpha_from_sigma_max(sigma_max, kappa_eff=1e3):
    s2 = sigma_max.square()
    floor = torch.finfo(sigma_max.dtype).eps * torch.maximum(s2, torch.ones_like(s2))
    return torch.maximum((sigma_max / float(kappa_eff)).square(), floor)


def _relative_flat_norm(x, ref):
    x = x.reshape(x.shape[0], -1)
    ref = ref.reshape(ref.shape[0], -1)
    eps = torch.finfo(ref.real.dtype).eps
    return torch.linalg.vector_norm(x, dim=1) / torch.linalg.vector_norm(ref, dim=1).clamp_min(eps)


def _iterative_feature_reliable(z, anchor, max_abs=1e3, max_deviation_ratio=1e2):
    if not torch.isfinite(z).all():
        return False, "nonfinite_feature"
    if max_abs is not None and z.abs().max() > max_abs:
        return False, "feature_abs"
    if max_deviation_ratio is not None:
        zf, af = z.reshape(z.shape[0], -1), anchor.reshape(anchor.shape[0], -1)
        if zf.shape != af.shape:
            return False, "shape_mismatch"
        dn = torch.linalg.vector_norm(zf - af, dim=1)
        an = torch.linalg.vector_norm(af, dim=1)
        floor = 1e-6 * math.sqrt(af.shape[1])
        if not torch.all((an <= floor) | (dn <= max_deviation_ratio * an)):
            return False, "feature_deviation"
    return True, "ok"


def _check_linearized_step_reliability(
    delta, J, residual, max_abs=1e3, max_norm_amp=1e4, opt_tol=1e-3):
    if not torch.isfinite(delta).all():
        return False, "nonfinite_delta"
    if max_abs is not None and delta.abs().max() > max_abs:
        return False, "delta_abs"

    dtype = torch.complex128 if torch.is_complex(J) else torch.float64
    J, d, r = J.detach().to(dtype), delta.detach().to(dtype), residual.detach().to(dtype)
    tiny = torch.finfo(torch.float64).eps
    Jn = torch.linalg.norm(J, ord="fro").real.clamp_min(tiny)
    dn = torch.linalg.vector_norm(d).real
    rn = torch.linalg.vector_norm(r).real.clamp_min(tiny)

    if max_norm_amp is not None:
        amp = (rn<1e-4)|(Jn * dn / rn)
        if not torch.isfinite(amp) or amp > max_norm_amp:
            return False, "norm_amp"

    lr = J @ d - r
    grad = J.mH @ lr
    rel_opt = torch.linalg.vector_norm(grad).real / (Jn * (Jn * dn + rn)).clamp_min(tiny)
    if not torch.isfinite(rel_opt) or rel_opt > opt_tol:
        return False, "lstsq_opt"
    return True, "ok"


def _check_mdp_step_reliability(delta, J, residual, anchor_offset, alpha, opt_tol=1e-2):
    if not torch.isfinite(delta).all():
        return False, "nonfinite_regularized_delta"

    dtype = torch.complex128 if torch.is_complex(J) else torch.float64
    J, d, r, e = [v.detach().to(dtype) for v in (J, delta, residual, anchor_offset)]
    a = torch.as_tensor(alpha, dtype=torch.float64, device=J.device).to(dtype)
    grad = J.mH @ (J @ d - r) + a * (e + d)

    tiny = torch.finfo(torch.float64).eps
    Jn = torch.linalg.norm(J, ord="fro").real
    denom = (
        (Jn.square() + a.real) * torch.linalg.vector_norm(d).real
        + Jn * torch.linalg.vector_norm(r).real
        + a.real * torch.linalg.vector_norm(e).real
    ).clamp_min(tiny)
    rel = torch.linalg.vector_norm(grad).real / denom
    if not torch.isfinite(rel) or rel > opt_tol:
        return False, "regularized_optimality"
    return True, "ok"


def _mdp_centered_regularized_lstsq(
    J, residual, anchor_offset, kappa_eff=1e3, alpha=None, alpha_lower_bound=0.0,):
    """min_d ||Jd-r||^2 + alpha ||anchor_offset+d||^2."""
    orig_dtype = residual.dtype
    dtype = torch.complex128 if torch.is_complex(J) else torch.float64
    J, r, e = J.detach().to(dtype), residual.detach().to(dtype), anchor_offset.detach().to(dtype)
    n = J.shape[1]

    if alpha is None:
        try:
            sigma_max = torch.linalg.svdvals(J)[0].real
        except RuntimeError:
            sigma_max = torch.linalg.norm(J, ord="fro").real
        a = _tikhonov_alpha_from_sigma_max(sigma_max, kappa_eff)
        a = torch.maximum(a, torch.as_tensor(alpha_lower_bound, dtype=a.dtype, device=J.device))
    else:
        a = torch.as_tensor(alpha, dtype=torch.float64, device=J.device)
        if a <= 0:
            raise ValueError(f"alpha must be positive, got {a.item()}")

    eye = torch.eye(n, device=J.device, dtype=dtype)
    lhs = J.mH @ J + a.to(dtype) * eye
    rhs = J.mH @ r - a.to(dtype) * e
    try:
        d = torch.linalg.solve(lhs, rhs)
    except RuntimeError:
        d = torch.linalg.lstsq(lhs, rhs.unsqueeze(-1)).solution.squeeze(-1)
    return d.to(orig_dtype), a


def _backtracking_step(
    z, delta, init, target, func, current_err, damping=0.8, max_step_norm=10.0,
    max_feature_abs=1e3, max_deviation_ratio=1e2, trials=8):
    dn = torch.linalg.vector_norm(delta)
    if not torch.isfinite(dn) or dn <= 0:
        return z, False, 0.0, float(current_err)

    if max_step_norm is not None and dn > max_step_norm:
        delta = delta * (max_step_norm / dn)

    step = float(damping)
    for _ in range(trials):
        candidate = z + step * delta.reshape_as(z)
        ok, _ = _iterative_feature_reliable(
            candidate, init, max_abs=max_feature_abs,
            max_deviation_ratio=max_deviation_ratio)
        if ok:
            with torch.no_grad():
                err = torch.linalg.vector_norm((target - func(candidate)).reshape(-1))
            if torch.isfinite(err) and err < current_err:
                return candidate.detach(), True, step, float(err.detach().cpu())
        step *= 0.8
    return z, False, 0.0, float(current_err)


def _linearized_reverse_map(
    target, init, func, steps=20, damping=0.8, max_step_norm=10.0, tol=1e-6,
    reliability_opt_tol=1e-3, max_delta_abs=1e3, max_norm_amp=1e2,
    max_feature_abs=1e3, max_deviation_ratio=1e2, kappa_eff=1e3,
    tikhonov_alpha=None, return_info=False,
):
    """Explicit-Jacobian Gauss--Newton with MDP-centered Tikhonov fallback."""
    z = init.detach().clone()
    target = target.detach().to(z)
    shape = z.shape
    tiny = torch.finfo(z.real.dtype).eps
    history, methods, reliability, step_scales = [], [], [], []

    for _ in range(steps):
        zf = z.reshape(-1).detach().requires_grad_(True)

        def flat_func(x):
            return func(x.reshape(shape)).reshape(-1)

        y = flat_func(zf)
        residual = (target.reshape(-1) - y).detach()
        err = torch.linalg.vector_norm(residual)
        rel = err / torch.linalg.vector_norm(target.reshape(-1)).clamp_min(tiny)
        history.append(float(rel.detach().cpu()))
        if not torch.isfinite(rel) or rel <= tol:
            break

        J = torch.autograd.functional.jacobian(flat_func, zf, vectorize=True).detach()
        try:
            delta = torch.linalg.lstsq(J, residual.unsqueeze(-1)).solution.squeeze(-1)
            ok, reason = _check_linearized_step_reliability(
                delta, J, residual, max_abs=max_delta_abs,
                max_norm_amp=max_norm_amp, opt_tol=reliability_opt_tol,
            )
        except RuntimeError:
            delta, ok, reason = torch.zeros_like(zf), False, "lstsq_failure"

        if ok:
            candidate, accepted, scale, _ = _backtracking_step(
                z, delta, init, target, func, err, damping, max_step_norm,
                max_feature_abs, max_deviation_ratio,
            )
            if accepted:
                z = candidate
                methods.append("lstsq")
                reliability.append((True, "ok"))
                step_scales.append(scale)
                continue
            ok, reason = False, "nonlinear_descent"

        e = zf.detach() - init.reshape(-1).detach()
        b_anchor = residual + J @ e
        allowed = []
        if max_feature_abs is not None:
            margin = float(max_feature_abs) - float(init.abs().max())
            if margin > 0:
                allowed.append(margin)
        if max_deviation_ratio is not None:
            init_norm = float(torch.linalg.vector_norm(init.reshape(-1)))
            if init_norm > 1e-12:
                allowed.append(float(max_deviation_ratio) * init_norm)
        alpha_lb = (
            float(torch.linalg.vector_norm(b_anchor) / (2.0 * min(allowed))) ** 2
            if allowed and min(allowed) > 0 else 0.0
        )

        delta, used_alpha = _mdp_centered_regularized_lstsq(
            J, residual, e, kappa_eff=kappa_eff,
            alpha=tikhonov_alpha, alpha_lower_bound=alpha_lb,
        )
        reg_ok, reg_reason = _check_mdp_step_reliability(
            delta, J, residual, e, used_alpha,
            opt_tol=max(reliability_opt_tol, 1e-2),
        )
        accepted, scale = False, 0.0
        if reg_ok:
            candidate, accepted, scale, _ = _backtracking_step(
                z, delta, init, target, func, err, damping, max_step_norm,
                max_feature_abs, max_deviation_ratio,
            )
            if accepted:
                z = candidate
            else:
                reg_reason = "nonlinear_descent"

        methods.append(f"mdp_tikhonov({reason},alpha={float(used_alpha):.3e})")
        reliability.append((accepted, "ok" if accepted else reg_reason))
        step_scales.append(scale if accepted else 0.0)
        if not accepted:
            break

    info = {
        "history": history, "methods": methods, "reliability": reliability,
        "step_scales": step_scales,
    }
    return (z.detach(), info) if return_info else (z.detach(), history)


def _vjp_reverse_map(
    target, init, func, steps=20, lr=1e-2, max_step_norm=10.0, tol=1e-6,
    max_deviation_ratio=1e2, return_info=False,
):
    """Matrix-free VJP solve with MDP-centered projection."""
    target = target.detach()
    init = init.detach().to(target)
    z = init.clone().requires_grad_(True)

    eps = torch.finfo(target.real.dtype).eps
    target_norm = torch.linalg.vector_norm(
        target.reshape(target.shape[0], -1), dim=1).clamp_min(eps)

    init_flat = init.reshape(init.shape[0], -1)
    init_norm = torch.linalg.vector_norm(init_flat, dim=1)
    ref_floor = 1e-4 * math.sqrt(init_flat.shape[1])
    radius = None if max_deviation_ratio is None else \
        float(max_deviation_ratio) * init_norm.clamp_min(ref_floor)

    opt = torch.optim.Adam([z], lr=lr)
    history, step_norms = [], []
    best_z, best_rel = init.clone(), float("inf")

    for _ in range(steps):
        prev = z.detach().clone()
        opt.zero_grad(set_to_none=True)

        residual = func(z) - target
        rel = torch.linalg.vector_norm(
            residual.reshape(residual.shape[0], -1), dim=1) / target_norm
        rel_mean = rel.mean()
        history.append(float(rel_mean.detach().cpu()))

        if torch.isfinite(rel_mean) and rel_mean < best_rel:
            best_rel, best_z = float(rel_mean.detach().cpu()), z.detach().clone()
        if not torch.isfinite(rel_mean) or rel_mean <= tol:
            break

        rel.square().mean().backward()
        if z.grad is None or not torch.isfinite(z.grad).all():
            break

        opt.step()

        with torch.no_grad():
            # Step-size constraint.
            delta = z - prev
            norms = torch.linalg.vector_norm(
                delta.reshape(delta.shape[0], -1), dim=1)
            step_norms.append(float(norms.max().detach().cpu()))

            if max_step_norm is not None:
                scale = (max_step_norm / norms.clamp_min(1e-12)).clamp(max=1.0)
                z.copy_(
                    prev + delta * scale.view(-1, *([1] * (delta.ndim - 1)))
                )

            # MDP-centered projection.
            if radius is not None:
                dev = z - init
                dev_norm = torch.linalg.vector_norm(
                    dev.reshape(dev.shape[0], -1), dim=1)
                scale = (radius / dev_norm.clamp_min(1e-12)).clamp(max=1.0)
                z.copy_(
                    init + dev * scale.view(-1, *([1] * (dev.ndim - 1)))
                )

            if not torch.isfinite(z).all():
                z.copy_(prev)
                break

    with torch.no_grad():
        rel = _relative_flat_norm(func(z) - target, target).mean()
        if torch.isfinite(rel) and rel < best_rel:
            best_rel, best_z = float(rel.detach().cpu()), z.detach().clone()

    info = {
        "history": history,
        "methods": ["vjp_projected_adam"] * max(0, len(history) - 1),
        "step_norms": step_norms,
        "best_rel_error": best_rel,
    }
    return (best_z, info) if return_info else (best_z, history)



def _estimate_jacobian_elems(init, target):
    return int(init.numel()) * int(target.numel())


def _choose_reverse_method(init, target, prefer="auto", jacobian_elem_limit=5e7):
    if prefer in ("linearized", "vjp"):
        return prefer
    if prefer != "auto":
        raise ValueError("method must be 'auto', 'linearized', or 'vjp'")
    return "linearized" if _estimate_jacobian_elems(init, target) <= jacobian_elem_limit else "vjp"

def _reverse_map_auto(
    target, init, func, steps=20, damping=0.8, max_step_norm=10.0, tol=1e-6,
    method="auto", jacobian_elem_limit=5e7, kappa_eff=1e3,
    tikhonov_alpha=None, max_feature_abs=1e3, max_deviation_ratio=1e2, return_info=False,
):
    selected = _choose_reverse_method(init, target, method, jacobian_elem_limit)
    requested, fallback = method, None
    jacobian_elems = _estimate_jacobian_elems(init, target)

    if selected == "linearized":
        try:
            z, info = _linearized_reverse_map(
                target, init, func, steps, damping, max_step_norm, tol,
                kappa_eff=kappa_eff,
                tikhonov_alpha=tikhonov_alpha,
                max_feature_abs=max_feature_abs,
                max_deviation_ratio=max_deviation_ratio,
                return_info=True,
            )
            if info["history"] and not math.isfinite(info["history"][-1]):
                raise RuntimeError("non-finite linearized reconstruction")
        except RuntimeError as exc:
            fallback = f"linearized_to_vjp: {exc}"
            if init.is_cuda:
                torch.cuda.empty_cache()
            selected = "vjp"

    if selected == "vjp":
        z, info = _vjp_reverse_map(
            target, init, func,
            steps=steps,
            lr=min(float(damping), 1e-2),
            max_step_norm=max_step_norm,
            tol=tol,
            max_deviation_ratio=max_deviation_ratio,
            return_info=True,
        )

        ok, reason = _iterative_feature_reliable(
            z, init,
            max_abs=max_feature_abs,
            max_deviation_ratio=max_deviation_ratio,
        )
        if not ok:
            z = init.detach().clone()
            info["constraint_fallback"] = f"anchor_return({reason})"

    info["requested_method"] = requested
    info["selected_method"] = selected
    info["jacobian_elems"] = jacobian_elems
    info["jacobian_elem_limit"] = jacobian_elem_limit
    if fallback is not None:
        info["fallback"] = fallback

    return (z, info) if return_info else (z, info["history"])
def _first_tensor(obj):
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, (tuple, list)):
        for x in obj:
            out = _first_tensor(x)
            if out is not None:
                return out
    if isinstance(obj, dict):
        for x in obj.values():
            out = _first_tensor(x)
            if out is not None:
                return out
    return None


def _call_module_tensor(module, x):
    out = _first_tensor(module(x))
    if out is None:
        raise RuntimeError(f"{module.__class__.__name__} returned no tensor")
    return out


def composite_reverseCom(
    front_fea, back_fea_star, _block, iter=20, damping=0.8, max_step_norm=10.0,
    tol=1e-6, method="auto", jacobian_elem_limit=5e7, kappa_eff=1e3,
    tikhonov_alpha=None, max_feature_abs=1e3,
    max_deviation_ratio=1e2, return_info=False, func=None,
):
    """Reverse an arbitrary differentiable composite block y = Phi(x)."""
    was_training = _block.training
    req_grad = [p.requires_grad for p in _block.parameters()]
    _block.eval()
    for p in _block.parameters():
        p.requires_grad_(False)

    x0 = front_fea.detach()
    target = back_fea_star.detach().to(x0)
    func = (lambda x: _call_module_tensor(_block, x)) if func is None else func

    try:
        x, info = _reverse_map_auto(
            target, x0, func, steps=iter, damping=damping,
            max_step_norm=max_step_norm, tol=tol, method=method,
            jacobian_elem_limit=jacobian_elem_limit, kappa_eff=kappa_eff,
            tikhonov_alpha=tikhonov_alpha, max_feature_abs=max_feature_abs,
            max_deviation_ratio=max_deviation_ratio,return_info=True,
        )
        if return_info:
            with torch.no_grad():
                info["final_rel_error"] = _relative_flat_norm(func(x) - target, target)
            return x, info
        return x
    finally:
        for p, r in zip(_block.parameters(), req_grad):
            p.requires_grad_(r)
        _block.train(was_training)
