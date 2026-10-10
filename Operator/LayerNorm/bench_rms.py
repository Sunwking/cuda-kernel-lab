import os, time, torch
here = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", os.path.join(here, ".torch_ext"))

import torch.utils.cpp_extension as ce
ce.get_compiler_abi_compatibility_and_version = lambda compiler: (True, "19.40.0")

ext = ce.load(
    name="ln_rms_block",
    sources=[os.path.join(here, "rmsnorm_block.cu")],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-Xcompiler", "/Zc:preprocessor"],
    verbose=False,
)

torch.manual_seed(0)
x = torch.randn(1024, 4096, device="cuda")
w = torch.randn(4096, device="cuda")
eps = 1e-6

ref = torch.nn.functional.rms_norm(x, [4096], w, eps)
out = ext.rmsnorm_v1(x, w, eps)
print("allclose:", torch.allclose(out, ref, atol=1e-4, rtol=1e-4))
print("max err :", (out - ref).abs().max().item())

def bench(fn, iters=100):
    for _ in range(10): fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1e6

print("torch     : %.1f us" % bench(lambda: torch.nn.functional.rms_norm(x, [4096], w, eps)))
print("rmsnorm_v1: %.1f us" % bench(lambda: ext.rmsnorm_v1(x, w, eps)))
