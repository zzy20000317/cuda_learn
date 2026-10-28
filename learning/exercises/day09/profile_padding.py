"""Only two 1024x1024 kernels, for Nsight counter collection (not timing)."""
from pathlib import Path
import torch
from torch.utils.cpp_extension import load
HERE=Path(__file__).resolve().parent
lib=load(name='learning_day09_padding',sources=[str(HERE/'transpose_padding.cu')],
         extra_cuda_cflags=['-O3','-lineinfo'],extra_cflags=['-O3'])
a=torch.randn(1024,1024,device='cuda');b=torch.empty_like(a)
for version in [0,1]:
    lib.transpose(a,b,version)
    torch.cuda.synchronize()
    torch.testing.assert_close(b,a.t(),rtol=0,atol=0)
print('PASS: profile kernels 32x32 and 32x33')
