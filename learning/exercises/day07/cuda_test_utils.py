from pathlib import Path
import csv
import json
import statistics
import torch

def time_ms(fn, warmup=20, repeats=50, trials=5):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    times=[]
    for _ in range(trials):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(repeats): fn()
        b.record(); b.synchronize()
        times.append(a.elapsed_time(b)/repeats)
    return statistics.median(times)

def must_reject(fn):
    try: fn()
    except (RuntimeError,ValueError): return
    raise AssertionError('invalid input was silently accepted')

def save_results(here, rows, checks, extra=None):
    p=Path(here)/'results'; p.mkdir(exist_ok=True)
    metadata=dict(gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
                  torch=torch.__version__,cuda=torch.version.cuda,
                  timing='CUDA events; warm cache; 20 warmup; median of 5 x 50 calls; excludes JIT and input initialization',
                  checks=checks, rows=rows)
    if extra: metadata.update(extra)
    (p/'results.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False))
    if rows:
        with (p/'benchmark.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('PASS:',checks)
    for r in rows: print(r)
