from pathlib import Path
import sys
import csv
import torch
from torch.utils.cpp_extension import load
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'day08'))
from test_transpose import check_versions,benchmark_versions
from cuda_test_utils import save_results

def main():
    torch.manual_seed(42)
    lib=load(name='learning_day09_padding',sources=[str(HERE/'transpose_padding.cu')],
             extra_cuda_cflags=['-O3','-lineinfo'],extra_cflags=['-O3'])
    checks=check_versions(lib,[0,1]);rows=benchmark_versions(lib,['shared_32x32','padded_32x33'])
    for i in range(0,len(rows),2):
        a,b=rows[i:i+2]
        for r in [a,b]:r['speedup_vs_unpadded']=a['custom_ms']/r['custom_ms']
    save_results(HERE,rows,checks,dict(allocation='preallocated output',
                                     shared_bytes=dict(unpadded=4096,padded=4224),
                                     bandwidth='8 * rows * cols / seconds; effective traffic, not physical DRAM bandwidth'))
    p=HERE/'results/bank_mapping.csv'
    with p.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['lane','column','bank_stride32','bank_stride33'])
        for col in [0,1,7,31]:
            for lane in range(32):w.writerow([lane,col,(lane*32+col)%32,(lane*33+col)%32])
    from draw_bank_map import draw
    draw(HERE/'results/bank_mapping.svg')
    print('PASS: bank mapping exported; stride32 -> same bank; stride33 -> distinct banks')
if __name__=='__main__':main()
