
import gc
import json
import math
import time
from pathlib import Path
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "imagecls"))
from rvs_cpt import convo_reverseCom, build_conv2d_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
DEVICE = torch.device("cuda")

def make_weight_with_condition(
    c_out,
    c_in,
    kernel_size,
    condition,
    dtype=None,
    rank_mode="full",
    effective_rank=None,
    min_singular=0.0,
    weight_scale=1.0,
):
    """Create a conv kernel with controlled flattened-kernel spectrum.

    The flattened kernel is [c_out, c_in * kernel_size * kernel_size].
    Its largest singular value is kept at weight_scale, and ill-conditioning
    is introduced by shrinking weak directions instead of amplifying strong
    directions. This avoids exploding weight values for large condition.
    """
    if dtype is None:
        dtype = torch.float32
    if condition < 1:
        raise ValueError(f"condition must be >= 1, got {condition}")

    cols = c_in * kernel_size * kernel_size
    max_rank = min(c_out, cols)
    if effective_rank is None:
        effective_rank = max_rank
    effective_rank = max(1, min(int(effective_rank), max_rank))

    q_left, _ = torch.linalg.qr(torch.randn(c_out, max_rank, device=DEVICE, dtype=torch.float64))
    q_right, _ = torch.linalg.qr(torch.randn(cols, max_rank, device=DEVICE, dtype=torch.float64))

    singulars = torch.zeros(max_rank, device=DEVICE, dtype=torch.float64)
    if effective_rank == 1:
        singulars[0] = 1.0
    else:
        singulars[:effective_rank] = torch.logspace(
            0.0,
            -math.log10(condition),
            effective_rank,
            device=DEVICE,
            dtype=torch.float64,
        )

    if rank_mode == "full":
        if effective_rank < max_rank:
            fill = max(float(min_singular), torch.finfo(torch.float64).eps)
            singulars[effective_rank:] = fill
    elif rank_mode == "near_rank_deficient":
        if effective_rank < max_rank:
            fill = max(float(min_singular), 1.0 / (float(condition) * 100.0))
            singulars[effective_rank:] = fill
    elif rank_mode == "rank_deficient":
        pass
    else:
        raise ValueError(f"unknown rank_mode={rank_mode!r}")

    singulars = singulars * float(weight_scale)
    weight_2d = q_left @ torch.diag(singulars) @ q_right.T
    return weight_2d.reshape(c_out, c_in, kernel_size, kernel_size).to(dtype)


def weight_condition_number(weight):
    s = torch.linalg.svdvals(weight.detach().reshape(weight.shape[0], -1).to(torch.float32))
    return float((s.max() / s.min().clamp_min(1e-30)).detach().cpu())

def conv_matrix_condition_number(layer, x_shape, max_dim=8192):
    n = math.prod(x_shape[1:])
    y_h = x_shape[2] + 2 * layer.padding[0] - layer.weight.shape[-2] + 1
    y_w = x_shape[3] + 2 * layer.padding[0] - layer.weight.shape[-1] + 1
    m = layer.out_channels * y_h * y_w
    if max(m, n) > max_dim:
        # print(
        #     f"[SKIP] conv_matrix_condition_number x_shape={x_shape} "
        #     f"A_shape=({m}, {n}) exceeds max_dim={max_dim}",
        #     flush=True,
        # )
        return None
    try:
        A = build_conv2d_matrix(layer, x_shape).to(torch.float64)
        return float(torch.linalg.cond(A).detach().cpu())
    except RuntimeError as exc:
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        # print(f"[FAILED] conv_matrix_condition_number x_shape={x_shape} A_shape=({m}, {n}) error={exc!r}", flush=True)
        return None


def cuda_sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def cuda_memory_mb(device):
    if device.type != "cuda":
        return {
            "allocated": None,
            "reserved": None,
            "max_allocated": None,
            "max_reserved": None,
        }
    return {
        "allocated": torch.cuda.memory_allocated(device) / 1024**2,
        "reserved": torch.cuda.memory_reserved(device) / 1024**2,
        "max_allocated": torch.cuda.max_memory_allocated(device) / 1024**2,
        "max_reserved": torch.cuda.max_memory_reserved(device) / 1024**2,
    }


def run_one(method, x, y_target, conv, repeats, device, context=None, regu=True):
    y_shape = tuple(y_target.shape)

    try:
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        cuda_sync(device)
        baseline_mem = cuda_memory_mb(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        start = time.perf_counter()
        x_new = None
        for _ in range(repeats):
            x_new = convo_reverseCom(x, y_target, conv, method=method, regu=regu)
        cuda_sync(device)
        elapsed = time.perf_counter() - start
        peak_mem = cuda_memory_mb(device)

        with torch.no_grad():
            y_new = conv(x_new.to(torch.float32))
            diff = y_new - y_target
            consistency_l2 = torch.linalg.vector_norm(diff).item()
            target_l2 = torch.linalg.vector_norm(y_target).item()
            distance_l2 = torch.linalg.vector_norm((x_new - x).reshape(x.shape[0], -1), dim=1).mean().item()
            front_l2 = torch.linalg.vector_norm(x.reshape(x.shape[0], -1), dim=1).mean().item()
            max_abs = diff.abs().max().item()
            
        return {
            "status": 'ok',
            "time_total_s": elapsed,
            "time_per_call_s": elapsed / repeats,
            "cuda_baseline_allocated_mb": baseline_mem["allocated"],
            "cuda_baseline_reserved_mb": baseline_mem["reserved"],
            "cuda_peak_mem_mb": peak_mem["max_allocated"],
            "cuda_peak_reserved_mb": peak_mem["max_reserved"],
            "cuda_end_allocated_mb": peak_mem["allocated"],
            "cuda_end_reserved_mb": peak_mem["reserved"],
            "cuda_algorithm_peak_mem_mb": (
                None
                if baseline_mem["allocated"] is None or peak_mem["max_allocated"] is None
                else max(0.0, peak_mem["max_allocated"] - baseline_mem["allocated"])
            ),
            "cuda_algorithm_peak_reserved_mb": (
                None
                if baseline_mem["reserved"] is None or peak_mem["max_reserved"] is None
                else max(0.0, peak_mem["max_reserved"] - baseline_mem["reserved"])
            ),
            "consistency_l2": consistency_l2,
            "consistency_rel_l2": consistency_l2 / max(target_l2, 1e-30),
            "consistency_max_abs": max_abs,
            "front_distance_l2": distance_l2,
            "front_distance_rel_l2": distance_l2 / max(front_l2, 1e-30),
        }
    except RuntimeError as exc:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        context_msg = "" if context is None else " " + " ".join(f"{k}={v}" for k, v in context.items())
        print(f"[FAILED] method={method}{context_msg} error={exc!r}", flush=True)
        return {"status": "failed", "error": repr(exc)}


def exact_matrix_skip_reason(layer, x_shape, max_dim=8192):
    n = math.prod(x_shape[1:])
    y_h = x_shape[2] + 2 * layer.padding[0] - layer.weight.shape[-2] + 1
    y_w = x_shape[3] + 2 * layer.padding[1] - layer.weight.shape[-1] + 1
    m = layer.out_channels * y_h * y_w
    if max(m, n) > max_dim:
        return f"exact_matrix dense A shape=({m}, {n}) exceeds max_dim={max_dim}"
    return None


def append_json_records(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    old = []
    if path.exists():
        old = json.loads(path.read_text())
    old.extend(records)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(old, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)
    

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    OUTPUT_JSON = Path("effi_stab_exp/reverse_conv_benchmark_results_0813_effi.json")

    SIZES = [16, 32, 64, 128, 256]
    REPEATS = 500
    BATCH_SIZE = 2
    C_IN = 2
    KERNEL_SIZE = 5
    USE_BIAS = False
    REGU_MODE = [False]
    RANK_MODES = ["full"]
    EFFECTIVE_RANK_RATIOS = [1.0]
    WEIGHT_SCALE = 1.0
    NOISE_SCALE = 1.0

    METHODS = ("exact_matrix", "cg", "fft_oldbound", "fft_pad")
    records = []
    for size in SIZES:
        for padding in [0]:
            for c_out in [4,1]:
                for regu in REGU_MODE:
                    max_kernel_rank = min(c_out, C_IN * KERNEL_SIZE * KERNEL_SIZE)
                    for rank_mode in RANK_MODES:
                        for effective_rank_ratio in EFFECTIVE_RANK_RATIOS:
                            effective_rank = max(1, min(max_kernel_rank, round(max_kernel_rank * effective_rank_ratio)))
                            for cond in [1.0]:
                                for seed in [0]:
                                    torch.cuda.empty_cache()
                                    torch.manual_seed(seed)
                                    x_shape = (BATCH_SIZE, C_IN, size, size)
                                    conv = nn.Conv2d(C_IN, c_out, KERNEL_SIZE, padding=(padding, padding), bias=USE_BIAS).to(DEVICE)
                                    # conv.weight.data = torch.rand(c_out, C_IN, KERNEL_SIZE, KERNEL_SIZE, device=DEVICE) * 2 - 1
                                    conv.weight.data = make_weight_with_condition(
                                        c_out,
                                        C_IN,
                                        KERNEL_SIZE,
                                        cond,
                                        rank_mode=rank_mode,
                                        effective_rank=effective_rank,
                                        weight_scale=WEIGHT_SCALE,
                                    )
                                    if conv.bias is not None:
                                        conv.bias.data.zero_()
                                    x = torch.rand(x_shape, dtype=torch.float32, device=DEVICE) * 2 - 1
                                    y = conv(x).detach()
                                    y_target = y + NOISE_SCALE * torch.rand_like(y)
                                    y_shape = tuple(y_target.shape)
                                    actual_weight_cond = weight_condition_number(conv.weight)
                                    actual_conv_cond = conv_matrix_condition_number(conv, x_shape)
                                    for method in METHODS:
                                        if method in ('exact_matrix', 'cg'):
                                            if torch.prod(torch.tensor(x.shape[1:])).item() >= torch.prod(torch.tensor(y_target.shape[1:])).item():
                                                type_alg = 'shrink'
                                            else: type_alg = 'expand'
                                        elif method in ("fft_oldbound", "fft_pad"):
                                            if x.shape[1] >= y_target.shape[1]:
                                                type_alg = 'shrink'
                                            else: type_alg = 'expand'
                                        else: raise ValueError("no such method")
                                        print(
                                            f"size={size} padding={padding} c2={c_out} case={type_alg} cond={cond} "
                                            f"rank_mode={rank_mode} effective_rank={effective_rank}/{max_kernel_rank} "
                                            f"noise_scale={NOISE_SCALE:g} seed={seed} method={method} regu={regu}",
                                            flush=True,
                                        )
                                        context = {
                                            "size": size,
                                            "padding": padding,
                                            "c2": c_out,
                                            "case": type_alg,
                                            "cond": cond,
                                            "rank_mode": rank_mode,
                                            "effective_rank": effective_rank,
                                            "seed": seed,
                                        }
                                        # skip_reason = (
                                        #     exact_matrix_skip_reason(conv, x_shape)
                                        #     if method == "exact_matrix"
                                        #     else None
                                        # )
                                        # if skip_reason is not None:
                                        #     print(f"[SKIPPED] method={method} {skip_reason}", flush=True)
                                        #     result = {"status": "skipped", "error": skip_reason}
                                        # else:
                                        result = run_one(method, x, y_target, conv, REPEATS, DEVICE, context, regu)
                                        record = {
                                            "method": method,
                                            "case_type": type_alg,
                                            "size": size,
                                            "padding": padding,
                                            "seed": seed,
                                            "repeats": REPEATS,
                                            "device": str(DEVICE),
                                            "requested_condition": cond,
                                            "rank_mode": rank_mode,
                                            "effective_rank": effective_rank,
                                            "max_kernel_rank": max_kernel_rank,
                                            "effective_rank_ratio": effective_rank_ratio,
                                            "weight_scale": WEIGHT_SCALE,
                                            "weight_condition": actual_weight_cond,
                                            "conv_matrix_condition": actual_conv_cond,
                                            "batch_size": BATCH_SIZE,
                                            "c_in": C_IN,
                                            "c_out": c_out,
                                            "kernel_size": KERNEL_SIZE,
                                            "x_shape": x_shape,
                                            "y_shape": y_shape,
                                            "use_regu":regu,
                                            **result,
                                        }
                                        records.append(record)
                                        append_json_records(OUTPUT_JSON, [record])
                                    del x, y, y_target, conv
                                    gc.collect()
                    print(f"^^^^^^^regu={regu}^^^^^^^^\n")
    print(f"saved {len(records)} new records to {OUTPUT_JSON}")
