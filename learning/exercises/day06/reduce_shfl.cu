#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <limits>

constexpr int WARP_SIZE = 32;
// 全部 32 lane 参与；无输入的 lane 贡献零，不提前退出。
template<typename T>
__device__ __forceinline__ T warpReduceSum(T v) {
#pragma unroll
    for(int offset=16;offset>0;offset>>=1) v+=__shfl_down_sync(0xffffffffu,v,offset);
    return v;
}
template<int BLOCK_SIZE>
__global__ void reduce_sum_shfl_kernel(const float* a,float* out,int64_t n) {
    int tid=threadIdx.x,lane=tid%32,warp=tid/32;
    constexpr int NUM_WARPS=BLOCK_SIZE/32;
    float sum=0;
    for(int64_t i=int64_t(blockIdx.x)*BLOCK_SIZE+tid;i<n;i+=int64_t(BLOCK_SIZE)*gridDim.x) sum+=a[i];
    sum=warpReduceSum(sum);
    __shared__ float warp_sums[NUM_WARPS];
    if(lane==0) warp_sums[warp]=sum;
    __syncthreads();
    if(warp==0) {
        float v=lane<NUM_WARPS?warp_sums[lane]:0.0f;
        v=warpReduceSum(v);
        if(lane==0) atomicAdd(out,v);
    }
}
torch::Tensor reduce_sum_shfl(torch::Tensor a,int64_t block,int64_t max_blocks) {
    TORCH_CHECK(a.is_cuda(),"must be CUDA");
    TORCH_CHECK(a.is_contiguous(),"must be contiguous");
    TORCH_CHECK(a.scalar_type()==at::kFloat,"must be float32");
    TORCH_CHECK(block==32||block==64||block==128||block==256||block==512||block==1024,
                "block must be 32/64/128/256/512/1024");
    TORCH_CHECK(max_blocks>0 && max_blocks<=std::numeric_limits<int>::max(),"max_blocks must be positive and <= INT_MAX");
    const c10::cuda::CUDAGuard guard(a.device());
    auto out=torch::zeros({1},a.options()); // 每次求和清零，atomicAdd 不能省略。
    int64_t n=a.numel();
    if(n==0) return out;
    int blocks=int(std::min((n+block-1)/block,max_blocks));
    auto stream=at::cuda::getCurrentCUDAStream();
#define LAUNCH(B) reduce_sum_shfl_kernel<B><<<blocks,B,0,stream>>>(a.data_ptr<float>(),out.data_ptr<float>(),n)
    switch(block) {
        case 32: LAUNCH(32);break; case 64: LAUNCH(64);break;
        case 128: LAUNCH(128);break; case 256: LAUNCH(256);break;
        case 512: LAUNCH(512);break; case 1024: LAUNCH(1024);break;
    }
#undef LAUNCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("reduce_sum_shfl",&reduce_sum_shfl);}
