#pragma once
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <algorithm>
inline void check_transpose(torch::Tensor a,torch::Tensor b) {
    TORCH_CHECK(a.is_cuda()&&b.is_cuda(),"CUDA required");
    TORCH_CHECK(a.device()==b.device(),"same device required");
    TORCH_CHECK(a.scalar_type()==at::kFloat&&b.scalar_type()==at::kFloat,"float32 required");
    TORCH_CHECK(a.dim()==2&&b.dim()==2,"2D tensors required");
    TORCH_CHECK(a.is_contiguous()&&b.is_contiguous(),"contiguous tensors required");
    TORCH_CHECK(b.size(0)==a.size(1)&&b.size(1)==a.size(0),"output must have transposed shape");
    if(a.numel()) {
        uintptr_t x=reinterpret_cast<uintptr_t>(a.data_ptr<float>()),y=reinterpret_cast<uintptr_t>(b.data_ptr<float>());
        uintptr_t bytes=a.numel()*sizeof(float);
        TORCH_CHECK(!(x<y+bytes && y<x+bytes),"input/output must not overlap");
    }
}
