#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <algorithm>

__global__ void add_review_kernel(const float* a,const float* b,float* c,int64_t n) {
    for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=int64_t(blockDim.x)*gridDim.x)
        c[i]=a[i]+b[i];
}
void check_tensor(torch::Tensor a) {
    TORCH_CHECK(a.is_cuda()&&a.scalar_type()==at::kFloat&&a.is_contiguous(),"contiguous CUDA float32 required");
}
void add_review(torch::Tensor a,torch::Tensor b,torch::Tensor c,int block) {
    check_tensor(a);check_tensor(b);check_tensor(c);
    TORCH_CHECK(a.device()==b.device()&&a.device()==c.device(),"same device required");
    TORCH_CHECK(a.sizes()==b.sizes()&&a.sizes()==c.sizes(),"same shape required");
    TORCH_CHECK(block==128||block==256||block==512,"block must be 128/256/512");
    const c10::cuda::CUDAGuard guard(a.device());int64_t n=a.numel();if(!n)return;
    int grid=int(std::min<int64_t>((n+block-1)/block,4096));
    add_review_kernel<<<grid,block,0,at::cuda::getCurrentCUDAStream()>>>(a.data_ptr<float>(),b.data_ptr<float>(),c.data_ptr<float>(),n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
__device__ float warp_sum(float v) {
#pragma unroll
    for(int d=16;d>0;d/=2)v+=__shfl_down_sync(0xffffffffu,v,d);
    return v;
}
// BUGGY=true 是可控的逻辑 bug：只输出 warp0 的局部和，无越界/竞争。
// 修复版收集所有 warp 的结果，经过 block 屏障后交给 warp0。
template<bool BUGGY>
__global__ void row_reduce(const float* a,float* out,int64_t cols) {
    int t=threadIdx.x,lane=t%32,warp=t/32;float v=0;
    for(int64_t c=t;c<cols;c+=256)v+=a[int64_t(blockIdx.x)*cols+c];
    v=warp_sum(v);
    if constexpr(BUGGY) {if(t==0)out[blockIdx.x]=v;}
    else {
        __shared__ float sums[8];if(lane==0)sums[warp]=v;__syncthreads();
        if(warp==0){v=lane<8?sums[lane]:0;v=warp_sum(v);if(lane==0)out[blockIdx.x]=v;}
    }
}
torch::Tensor row_sum(torch::Tensor a,bool buggy) {
    check_tensor(a);TORCH_CHECK(a.dim()==2,"2D input required");
    const c10::cuda::CUDAGuard guard(a.device());
    auto out=torch::empty({a.size(0)},a.options());if(a.size(0)==0)return out;
    auto stream=at::cuda::getCurrentCUDAStream();
    if(buggy)row_reduce<true><<<a.size(0),256,0,stream>>>(a.data_ptr<float>(),out.data_ptr<float>(),a.size(1));
    else row_reduce<false><<<a.size(0),256,0,stream>>>(a.data_ptr<float>(),out.data_ptr<float>(),a.size(1));
    C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("add",&add_review);m.def("row_sum",&row_sum);}
