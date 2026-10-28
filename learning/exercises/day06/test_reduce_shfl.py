from pathlib import Path
import sys
import torch
from torch.utils.cpp_extension import load
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'day07'))
from cuda_test_utils import time_ms,must_reject,save_results
lib=load(name='learning_reduce_shfl',sources=[str(HERE/'reduce_shfl.cu')],
         extra_cuda_cflags=['-O3','-lineinfo'],extra_cflags=['-O3'])

def main():
    torch.manual_seed(42)
    cases=0; max_error=0
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for block in [32,64,128,256,512,1024]:
            for cap in [1,17,512]:
                for n in [0,1,31,32,33,255,256,257,1025,65539]:
                    for exact in [True,False]:
                        a=torch.ones(n,device='cuda') if exact else torch.randn(n,device='cuda')
                        ref=a.double().sum();out=lib.reduce_sum_shfl(a,block,cap).squeeze().double()
                        error=(out-ref).abs().item();max_error=max(max_error,error)
                        tol=0 if exact else 1e-4+5e-5*max(abs(ref.item()),n**0.5)
                        assert error<=tol,(block,cap,n,error,tol)
                        cases+=1
        # Repeated calls must reset output, not accumulate previous invocations.
        a=torch.ones(257,device='cuda')
        for _ in range(3): assert lib.reduce_sum_shfl(a,256,17).item()==257
    stream.synchronize()
    good=torch.ones(8,device='cuda')
    for block in [0,-32,33,96,2048]:
        must_reject(lambda:lib.reduce_sum_shfl(good,block,512))
        must_reject(lambda:lib.reduce_sum_shfl(good[:0],block,512))
    for cap in [0,-1,2**31]: must_reject(lambda:lib.reduce_sum_shfl(good,256,cap))
    for a in [torch.ones(8),good.half(),torch.ones(16,device='cuda')[::2]]:
        must_reject(lambda:lib.reduce_sum_shfl(a,256,512))
    rows=[]
    for n in [1024,65536,1048576]:
        a=torch.randn(n,device='cuda');ref=a.double().sum()
        baseline=time_ms(lambda:a.sum())
        for block in [128,256,512]:
            actual=lib.reduce_sum_shfl(a,block,256).item()
            error=abs(actual-ref.item())
            assert error <= 1e-4+5e-5*max(abs(ref.item()),n**0.5)
            ms=time_ms(lambda:lib.reduce_sum_shfl(a,block,256))
            rows.append(dict(n=n,block=block,max_blocks=256,torch_ms=baseline,custom_ms=ms,speedup=baseline/ms,abs_error=error))
    save_results(HERE,rows,dict(numerical_cases=cases,nondefault_stream=True,invalid_inputs=True,
                               repeated_calls=True,max_abs_error=max_error),
                 dict(allocation='Both custom and torch allocate an output; custom includes zeroing and atomicAdd; timing is public API cost',
                      tolerance='exact ones; random: 1e-4 + 5e-5*max(abs(FP64 sum), sqrt(N))',
                      multigpu_tested=False))
if __name__=='__main__': main()
