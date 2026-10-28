import numpy as np
import torch
import flashinfer
from flashinfer.testing.utils import bench_gpu_time, get_l2_cache_size

assert callable(get_l2_cache_size)
a=torch.ones(1024, device='cuda')
b=torch.ones_like(a)
c=torch.empty_like(a)
times=bench_gpu_time(fn=lambda:torch.add(a,b,out=c), dry_run_iters=2,
                    repeat_iters=5, enable_cupti=False, use_cuda_graph=False,
                    cold_l2_cache=True)
assert len(times)>0 and np.isfinite(times).all() and (np.asarray(times)>0).all()
torch.testing.assert_close(c,a+b)
print('PASS: FlashInfer',flashinfer.__version__,'upstream benchmark API; median ms',float(np.median(times)))
