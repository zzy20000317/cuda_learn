"""Correctness first, then warm GPU event timing. Run via ./run python exercises/bench_add.py."""
from pathlib import Path
import json
import statistics
import torch
from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parents[1]
lib = load(name="learning_add", sources=[str(Path(__file__).with_name("add.cu"))],
           extra_cuda_cflags=["-O3", "-lineinfo"], extra_cflags=["-O3"])

def milliseconds(fn):
    for _ in range(20): fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(7):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(100): fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / 100)
    return statistics.median(samples)

def main():
    torch.manual_seed(42)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for n in [0, 1, 3, 31, 127, 129, 257, 1025, 65539]:
            for offset in [0, 1]:
                a = torch.randn(n+offset, device="cuda")[offset:]
                b = torch.randn_like(a)
                c = torch.empty_like(a)
                for block in [128,256,512]:
                    for vec in [False,True]:
                        lib.add(a,b,c,block,vec)
                        torch.testing.assert_close(c,a+b)
    stream.synchronize()
    print("PASS: empty, tails, unaligned input, block sizes, non-default stream")
    rows=[]
    for n in [1024,65536,1048576,4194304]:
        a=torch.randn(n,device="cuda"); b=torch.randn_like(a); c=torch.empty_like(a)
        baseline=milliseconds(lambda: torch.add(a,b,out=c))
        for block in [128,256,512]:
            for vec in [False,True]:
                ms=milliseconds(lambda: lib.add(a,b,c,block,vec))
                torch.testing.assert_close(c,a+b)
                row=dict(n=n,block=block,vectorized=vec,ms=ms,torch_ms=baseline,
                         speedup=baseline/ms,effective_GB_s=12*n/(ms*1e6))
                rows.append(row); print(row)
    ROOT.joinpath("results").mkdir(exist_ok=True)
    ROOT.joinpath("results/add-benchmark.json").write_text(json.dumps({
        "gpu":torch.cuda.get_device_name(),"torch":torch.__version__,"cuda":torch.version.cuda,
        "timing":"warm cache; CUDA events; 20 warmups; median of 7 x 100 launches; allocation excluded; effective bandwidth includes cache",
        "rows":rows},indent=2))
if __name__ == "__main__": main()
