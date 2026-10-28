from pathlib import Path
import sys
import torch
from torch.utils.cpp_extension import load
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'day07'))
from cuda_test_utils import time_ms,must_reject,save_results

def load_lib():
    return load(name='learning_day08_transpose',sources=[str(HERE/'transpose.cu')],
                extra_cuda_cflags=['-O3','-lineinfo'],extra_cflags=['-O3'])

def check_versions(lib,versions):
    cases=0;stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for rows,cols in [(0,0),(0,33),(33,0),(1,1),(1,65),(65,1),(31,33),(32,32),(256,512),(257,513),(513,257),(1025,33)]:
            # arange exposes indexing mistakes exactly; random covers normal data.
            for pattern in ['arange','random']:
                a=(torch.arange(rows*cols,device='cuda',dtype=torch.float32).reshape(rows,cols)
                   if pattern=='arange' else torch.randn(rows,cols,device='cuda'))
                for version in versions:
                    b=torch.full((cols,rows),float('nan'),device='cuda')
                    lib.transpose(a,b,version)
                    torch.testing.assert_close(b,a.t(),rtol=0,atol=0);cases+=1
        # A contiguous view may start at a nonzero storage offset.
        a=torch.arange(1+257*513,device='cuda',dtype=torch.float32)[1:].reshape(257,513)
        for v in versions:
            b=torch.empty(513,257,device='cuda');lib.transpose(a,b,v)
            torch.testing.assert_close(b,a.t(),rtol=0,atol=0);cases+=1
    stream.synchronize()
    a=torch.ones(32,32,device='cuda');b=torch.empty_like(a)
    for x,y in [(a.cpu(),b),(a.half(),b),(a.t(),b),(a,b[:,:16]),(a,a),(a.reshape(-1),b)]:
        must_reject(lambda:lib.transpose(x,y,versions[0]))
    must_reject(lambda:lib.transpose(a,b,99))
    return dict(exact_cases=cases,nondefault_stream=True,invalid_inputs=True,empty_dimensions=True,offset_view=True)

def benchmark_versions(lib,names,extra_shapes=()):
    rows=[]
    for m,n in [(32,32),(256,512),(257,513),(1024,1024),(2048,2048),*extra_shapes]:
        a=torch.randn(m,n,device='cuda');b=torch.empty(n,m,device='cuda')
        baseline=time_ms(lambda:b.copy_(a.t())) # both methods use preallocated output.
        copy_ms=time_ms(lambda:lib.copy(a,b)) if hasattr(lib,'copy') else None
        if copy_ms is not None:
            lib.copy(a,b);torch.testing.assert_close(b.reshape(-1),a.reshape(-1),rtol=0,atol=0)
        for v,name in enumerate(names):
            ms=time_ms(lambda:lib.transpose(a,b,v));torch.testing.assert_close(b,a.t(),rtol=0,atol=0)
            rows.append(dict(rows=m,cols=n,version=name,torch_ms=baseline,custom_ms=ms,
                             speedup_vs_torch=baseline/ms,copy_ms=copy_ms,effective_GB_s=8*m*n/(ms*1e6)))
    return rows

def main():
    torch.manual_seed(42);lib=load_lib();checks=check_versions(lib,[0,1,2])
    rows=benchmark_versions(lib,['naive_read','naive_write','shared_32x32'])
    save_results(HERE,rows,checks,dict(allocation='preallocated outputs for custom and torch copy of transposed view',
                                     bandwidth='2 * numel * 4 / seconds; includes cache effects; copy is not a transpose'))
if __name__=='__main__':main()
