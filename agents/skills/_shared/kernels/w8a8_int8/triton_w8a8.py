"""W8A8 int8 linear for any artifact saved in the `w8a8_int8` format.

The format: every quantized linear stores an int8 ``weight`` [out, in] (K-major)
and a float per-output-channel ``weight_scale`` [out, 1]; activations are
quantized at runtime, symmetric int8 per token. It is method-agnostic
(SmoothQuant writes it; so can any method that folds its transform into the
weights offline), and it is the layout `export_vllm.py` maps to
compressed-tensors `int-quantized`.

Kernels:

1. ``_act_quant_kernel`` / ``_act_quant_looped_kernel``: per-token absmax,
   fp32 scale and int8 codes in one pass (the looped variant for K > 4096).
   Codes and scales are bit-identical to rounding in the activation dtype the
   way torch does (clamp 1e-5, /127, round half to even).
2. ``_w8a8_gemm_kernel``: int8 x int8 -> int32 tensor-core GEMM with a fused
   epilogue, acc * act_scale[m] * weight_scale[n] (+ bias[n]) -> fp16/bf16.
   Any M, so decode and prefill share one kernel.

``w8a8_linear_reference`` is the dense fp16 path for CPU or for checking.
Needs torch and triton (bundled with CUDA torch); int8 ``tl.dot`` needs sm_80+.
Copy this file into an inference adapter, or import it by path; it has no other
dependencies.
"""
import time

import torch
import torch.nn.functional as F
import triton
import triton.language as tl



@triton.jit
def _act_quant_kernel(
    x_ptr, q_ptr, s_ptr, K, stride_xm, stride_qm,
    BLOCK_K: tl.constexpr,
):
    """Per-token absmax int8 quantization of one row of x (fp16/bf16)."""
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_K)
    mask = offs < K
    x_raw = tl.load(x_ptr + row * stride_xm + offs, mask=mask, other=0.0)
    act_ty = x_raw.dtype
    x = x_raw.to(tl.float32)
    amax = tl.max(tl.abs(x), axis=0)
    # Mirror torch on the activation dtype: clamp, cast, divide, cast.
    amax = tl.maximum(amax, 1e-5).to(act_ty).to(tl.float32)
    scale = tl.math.div_rn(amax, 127.0).to(act_ty).to(tl.float32)
    v = tl.math.div_rn(x, scale).to(act_ty).to(tl.float32)
    # Round half to even, like torch.round; |v| <= ~127.5 so v + 0.5 is exact.
    r = tl.floor(v + 0.5)
    tie = (r - v) == 0.5
    odd = (r - 2.0 * tl.floor(r * 0.5)) != 0.0
    r = tl.where(tie & odd, r - 1.0, r)
    r = tl.minimum(tl.maximum(r, -127.0), 127.0)
    tl.store(q_ptr + row * stride_qm + offs, r.to(tl.int8), mask=mask)
    tl.store(s_ptr + row, scale)


@triton.jit
def _act_quant_looped_kernel(
    x_ptr, q_ptr, s_ptr, K, stride_xm, stride_qm,
    BLOCK_K: tl.constexpr,
):
    """Same math as _act_quant_kernel for long rows: absmax pass, then quantize pass."""
    row = tl.program_id(0).to(tl.int64)
    x_row = x_ptr + row * stride_xm
    q_row = q_ptr + row * stride_qm
    offs = tl.arange(0, BLOCK_K)
    run_max = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        mask = (k0 + offs) < K
        x = tl.load(x_row + k0 + offs, mask=mask, other=0.0).to(tl.float32)
        run_max = tl.maximum(run_max, tl.abs(x))
    act_ty = x_ptr.dtype.element_ty
    amax = tl.max(run_max, axis=0)
    amax = tl.maximum(amax, 1e-5).to(act_ty).to(tl.float32)
    scale = tl.math.div_rn(amax, 127.0).to(act_ty).to(tl.float32)
    for k0 in range(0, K, BLOCK_K):
        mask = (k0 + offs) < K
        x = tl.load(x_row + k0 + offs, mask=mask, other=0.0).to(tl.float32)
        v = tl.math.div_rn(x, scale).to(act_ty).to(tl.float32)
        r = tl.floor(v + 0.5)
        tie = (r - v) == 0.5
        odd = (r - 2.0 * tl.floor(r * 0.5)) != 0.0
        r = tl.where(tie & odd, r - 1.0, r)
        r = tl.minimum(tl.maximum(r, -127.0), 127.0)
        tl.store(q_row + k0 + offs, r.to(tl.int8), mask=mask)
    tl.store(s_ptr + row, scale)


@triton.jit
def _w8a8_gemm_kernel(
    a_ptr, b_ptr, c_ptr, as_ptr, bs_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bn, stride_bk, stride_cm, stride_cn,
    HAS_BIAS: tl.constexpr,
    EVEN_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """C[M, N] = (A_int8[M, K] @ B_int8[N, K]^T) * a_scale[M] * b_scale[N] + bias[N]."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # Wrap out-of-range rows/cols onto valid memory; the store is masked.
    offs_am = offs_m % M
    offs_bn = offs_n % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_am[:, None].to(tl.int64) * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_bn[None, :].to(tl.int64) * stride_bn + offs_k[:, None] * stride_bk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            k_left = K - k * BLOCK_K
            a = tl.load(a_ptrs, mask=offs_k[None, :] < k_left, other=0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < k_left, other=0)
        acc = tl.dot(a, b, acc, out_dtype=tl.int32)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    a_scale = tl.load(as_ptr + offs_am).to(tl.float32)
    b_scale = tl.load(bs_ptr + offs_bn).to(tl.float32)
    out = acc.to(tl.float32) * a_scale[:, None] * b_scale[None, :]
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_bn).to(tl.float32)
        out = out + bias[None, :]
    c_ptrs = c_ptr + offs_m[:, None].to(tl.int64) * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, out.to(c_ptr.dtype.element_ty), mask=c_mask)


def quantize_activation_int8(x):
    """x [M, K] fp16/bf16 contiguous -> (int8 codes [M, K], fp32 per-token scale [M])."""
    m, k = x.shape
    q = torch.empty((m, k), dtype=torch.int8, device=x.device)
    scale = torch.empty((m,), dtype=torch.float32, device=x.device)
    if m == 0:
        return q, scale
    block_k = triton.next_power_of_2(int(k))
    args = (x, q, scale, int(k), int(x.stride(0)), int(q.stride(0)))
    if block_k <= 4096:
        # Whole row in registers: one read of x.
        _act_quant_kernel[(int(m),)](*args, BLOCK_K=block_k, num_warps=8)
    else:
        # Long rows: absmax pass + quantize pass over 4096-wide chunks; the
        # second read hits L2. Measured at copy bandwidth on sm_86.
        _act_quant_looped_kernel[(int(m),)](*args, BLOCK_K=4096, num_warps=8)
    return q, scale


def _gemm_config(m, n, k):
    """Default (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, num_warps, num_stages).

    Tuned on sm_86 (RTX 3090, see kernel.json `tuned_on`). On another compute
    capability, run `tune_config` on the model's shapes and pass the result as
    `config` to `w8a8_matmul`.
    """
    if m <= 16:
        return 16, 64, 256, 1, 4, 3
    if m <= 64:
        return 64, 64, 128, 8, 4, 4
    return 128, 128, 64, 8, 4, 4


def w8a8_matmul(q, a_scale, weight, weight_scale, bias, out_dtype, config=None):
    """int8 q [M, K] x int8 weight [N, K] -> out_dtype [M, N] with the dequant epilogue."""
    m, k = q.shape
    n = weight.shape[0]
    out = torch.empty((m, n), dtype=out_dtype, device=q.device)
    if m == 0:
        return out
    block_m, block_n, block_k, group_m, warps, stages = config or _gemm_config(m, n, k)
    grid = (triton.cdiv(int(m), block_m) * triton.cdiv(int(n), block_n),)
    _w8a8_gemm_kernel[grid](
        q, weight, out, a_scale, weight_scale, bias if bias is not None else a_scale,
        int(m), int(n), int(k),
        int(q.stride(0)), int(q.stride(1)),
        int(weight.stride(0)), int(weight.stride(1)),
        int(out.stride(0)), int(out.stride(1)),
        HAS_BIAS=bias is not None,
        EVEN_K=(int(k) % block_k == 0),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, GROUP_M=group_m,
        num_warps=warps, num_stages=stages,
    )
    return out


def w8a8_linear(x2d, weight, weight_scale, bias):
    """Real int8 W8A8 linear: Triton per-token quant, then int8 GEMM + epilogue."""
    q, a_scale = quantize_activation_int8(x2d)
    return w8a8_matmul(q, a_scale, weight, weight_scale, bias, x2d.dtype)


def w8a8_linear_reference(x, weight, weight_scale, bias):
    """fp16 dequant reference: same quantization, dense fp16 matmul. CPU-safe."""
    x_scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5) / 127
    x_q = torch.round(x / x_scale).clamp_(-127, 127).mul_(x_scale)
    w = weight.to(x.dtype).mul_(weight_scale.reshape(-1, 1).to(x.dtype))
    b = None if bias is None else bias.to(x.dtype)
    return F.linear(x_q, w, b)


CONFIG_CANDIDATES = (
    (128, 128, 64, 8, 4, 4),
    (128, 128, 64, 8, 8, 3),
    (128, 256, 64, 8, 8, 3),
    (64, 128, 128, 8, 4, 4),
    (64, 64, 128, 8, 4, 4),
    (16, 64, 256, 1, 4, 3),
)


def tune_config(m, n, k, *, device="cuda", candidates=CONFIG_CANDIDATES, iters=20):
    """Time each candidate tile config on random int8 inputs of this shape; return the fastest.

    Use once per (M, N, K) on a GPU whose compute capability is not in kernel.json
    `tuned_on`, then pass the result as `config` to `w8a8_matmul`.
    """
    q = torch.randint(-127, 128, (m, k), dtype=torch.int8, device=device)
    w = torch.randint(-127, 128, (n, k), dtype=torch.int8, device=device)
    a_scale = torch.rand(m, dtype=torch.float32, device=device)
    w_scale = torch.rand(n, 1, dtype=torch.float16, device=device)
    best, best_t = None, float("inf")
    for config in candidates:
        try:
            w8a8_matmul(q, a_scale, w, w_scale, None, torch.float16, config)
        except Exception:  # a tile the GPU cannot fit (shared memory, registers)
            continue
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(iters):
            w8a8_matmul(q, a_scale, w, w_scale, None, torch.float16, config)
        torch.cuda.synchronize(device)
        elapsed = (time.perf_counter() - start) / iters
        if elapsed < best_t:
            best, best_t = config, elapsed
    if best is None:
        raise RuntimeError(f"no candidate config ran for M={m} N={n} K={k}")
    return best
