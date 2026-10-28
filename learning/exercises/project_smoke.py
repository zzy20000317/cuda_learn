from pathlib import Path
import torch
from torch.utils.cpp_extension import load
root=Path(__file__).resolve().parents[2]
lib=load(name="vitamin_elementwise_smoke",sources=[str(root/"kernels/elementwise/elementwise.cu")],
 extra_cflags=["-O3","-std=c++17"],extra_cuda_cflags=["-O3","-std=c++17","-lineinfo",
 "-U__CUDA_NO_HALF_OPERATORS__","-U__CUDA_NO_HALF_CONVERSIONS__","-U__CUDA_NO_HALF2_OPERATORS__",
 "-U__CUDA_NO_BFLOAT16_CONVERSIONS__","--expt-relaxed-constexpr","--expt-extended-lambda"])
# Only aligned shapes for upstream vector kernels; learning/add.cu tests arbitrary tails.
for dtype in [torch.float32,torch.float16]:
 a=torch.randn(256,256,device="cuda",dtype=dtype); b=torch.randn_like(a); c=torch.empty_like(a)
 lib.elementwise_add(a,b,c)
 torch.testing.assert_close(c,a+b)
print("PASS: original project elementwise extension, FP32 and FP16")
