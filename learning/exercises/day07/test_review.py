from pathlib import Path
import torch
from torch.utils.cpp_extension import load
from cuda_test_utils import time_ms,must_reject,save_results
HERE=Path(__file__).resolve().parent
lib=load(name='learning_day07_review',sources=[str(HERE/'review.cu')],extra_cuda_cflags=['-O3','-lineinfo'],extra_cflags=['-O3'])
def main():
    torch.manual_seed(42);stream=torch.cuda.Stream();cases=0
    with torch.cuda.stream(stream):
        for n in [0,1,3,31,127,129,257,1025,65539]:
            for offset in [0,1]:
                a=torch.randn(n+offset,device='cuda')[offset:];b=torch.randn_like(a);c=torch.empty_like(a)
                for block in [128,256,512]:
                    lib.add(a,b,c,block);torch.testing.assert_close(c,a+b,rtol=0,atol=0);cases+=1
        witness=torch.ones(2,256,device='cuda')
        bad=lib.row_sum(witness,True);fixed=lib.row_sum(witness,False)
        assert torch.equal(bad,torch.full_like(bad,32))
        torch.testing.assert_close(fixed,witness.sum(1),rtol=0,atol=0)
        for shape in [(0,256),(3,0),(1,1),(4,31),(3,257),(7,1025)]:
            a=torch.randn(*shape,device='cuda');ref=a.double().sum(1)
            torch.testing.assert_close(lib.row_sum(a,False).double(),ref,rtol=1e-4,atol=1e-4)
    stream.synchronize()
    a=torch.ones(8,device='cuda');c=torch.empty_like(a)
    for args in [(a,a,c,33),(a,a[:3],c,256),(a.cpu(),a,c,256),(a.half(),a,c,256),(a,a,torch.empty(16,device='cuda')[::2],256)]:
        must_reject(lambda:lib.add(*args))
    rows=[]
    for n in [1024,65536,1048576]:
        a=torch.randn(n,device='cuda');b=torch.randn_like(a);c=torch.empty_like(a)
        baseline=time_ms(lambda:torch.add(a,b,out=c))
        for block in [128,256,512]:
            ms=time_ms(lambda:lib.add(a,b,c,block))
            rows.append(dict(n=n,block=block,torch_ms=baseline,custom_ms=ms,speedup=baseline/ms))
    save_results(HERE,rows,dict(add_cases=cases,row_reduce_shapes=6,nondefault_stream=True,invalid_inputs=True,
                               bug_expected=256,bug_observed=32,fixed_observed=256),dict(allocation='add uses preallocated output for both methods'))
if __name__=='__main__':main()
