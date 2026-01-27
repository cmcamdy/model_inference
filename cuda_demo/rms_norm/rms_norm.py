import time
from typing import Optional

import torch
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

# Load the CUDA kernel as a python module
lib = load(
    name="rms_norm_lib",
    sources=["main.cpp", "rms_norm.cu"],
    extra_cuda_cflags=[
        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
    ],
    extra_cflags=["-std=c++17"],
)


# un-fused naive rms norm
def naive_rms_norm(x: torch.Tensor, g: float):
    # y'=x/rms(x) 1/rms(x) = rsqrtf(sum(x^2)/K)
    # print(torch.sum(x**2, dim=1))
    s_rms = torch.rsqrt(torch.mean(x**2, dim=1, keepdim=True))
    # print(s_rms)
    y = (x * s_rms) * g
    return y


def run_benchmark(
    perf_func: callable,
    x: torch.Tensor,
    tag: str,
    out: Optional[torch.Tensor] = None,
    warmup: int = 100,
    iters: int = 1000,
    show_all: bool = False,
):
    g = 1.0
    if out is not None:
        out.fill_(0)
    if out is not None:
        for i in range(warmup):
            perf_func(x, out, g)
    else:
        for i in range(warmup):
            _ = perf_func(x, g)
    torch.cuda.synchronize()
    start = time.time()
    # iters
    if out is not None:
        for i in range(iters):
            perf_func(x, out, g)
    else:
        for i in range(iters):
            out = perf_func(x, g)
    torch.cuda.synchronize()
    end = time.time()
    total_time = (end - start) * 1000  # ms
    mean_time = total_time / iters
    out_info = f"out_{tag}"
    out_val = out.flatten().detach().cpu().numpy().tolist()[:3]
    out_val = [round(v, 8) for v in out_val]
    out_val = [f"{v:<12}" for v in out_val]
    print(f"{out_info:>17}: {out_val}, time:{mean_time:.8f}ms")
    if show_all:
        print(out)
    return out, mean_time


# print("-" * 85)
# N, K = 1, 512
# N, K = 4096, 512
# print(" " * 40 + f"N={N}, K={K}")
# x = torch.randn((N, K)).cuda().float().contiguous()
# out = torch.zeros_like(x).cuda().float().contiguous()
# out, mean_time = run_benchmark(lib.rms_norm_f32, x, "f32", out)
# out_n, mean_time_n = run_benchmark(naive_rms_norm, x, "f32_th")
# print(torch.max(torch.abs(out - out_n)))

print("-" * 85)
# N, K = 1, 1024
N, K = 4096, 8192
print(" " * 40 + f"N={N}, K={K}")
x = torch.randn((N, K)).cuda().float().contiguous()
out = torch.zeros_like(x).cuda().float().contiguous()
# out, mean_time = run_benchmark(lib.rms_norm_f32, x, "f32", out)
# out_x4, mean_time_x4 = run_benchmark(lib.rms_norm_f32x4, x, "f32", out)
out_x4v2, mean_time_x4v2 = run_benchmark(lib.rms_norm_f32x4v2, x, "f32x4v2", out)
out_n, mean_time_n = run_benchmark(naive_rms_norm, x, "f32_th")
print(torch.max(torch.abs(out_x4v2 - out_n)))
# for i in range(N):
#     # print(torch.max(torch.abs(out_x4[i] - out_n[i])))
#     # print(torch.max(torch.abs(out_x4v2[i] - out_n[i])))
#     # print(torch.max(torch.abs(out - out_n)))
#     print(out_x4v2[i] - out_n[i])
#     # print(out[i] - out_n[i])
#     # print(out[i])
#     # print(out_n[i])
