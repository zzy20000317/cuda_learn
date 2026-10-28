#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

// Exercise: explain indexing, tail handling, vector alignment and current stream.
__global__ void add_scalar(const float* a, const float* b, float* c, int64_t n) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += int64_t(blockDim.x) * gridDim.x) c[i] = a[i] + b[i];
}
__global__ void add_vector(const float* a, const float* b, float* c, int64_t n) {
  for (int64_t i = 4 * (int64_t(blockIdx.x) * blockDim.x + threadIdx.x);
       i < n; i += 4 * int64_t(blockDim.x) * gridDim.x) {
    if (i + 3 < n) {
      float4 x = reinterpret_cast<const float4*>(a)[i / 4];
      float4 y = reinterpret_cast<const float4*>(b)[i / 4];
      reinterpret_cast<float4*>(c)[i / 4] = make_float4(x.x+y.x,x.y+y.y,x.z+y.z,x.w+y.w);
    } else {
      for (int64_t j = i; j < n; ++j) c[j] = a[j] + b[j];
    }
  }
}
void add(torch::Tensor a, torch::Tensor b, torch::Tensor c, int threads, bool vectorized) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda() && c.is_cuda(), "CUDA tensors required");
  TORCH_CHECK(a.device()==b.device() && a.device()==c.device(), "same device required");
  TORCH_CHECK(a.scalar_type()==torch::kFloat32 && b.scalar_type()==torch::kFloat32 && c.scalar_type()==torch::kFloat32, "float32 required");
  TORCH_CHECK(a.sizes()==b.sizes() && a.sizes()==c.sizes(), "same shape required");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && c.is_contiguous(), "contiguous tensors required");
  TORCH_CHECK(threads==128 || threads==256 || threads==512, "threads must be 128/256/512");
  const c10::cuda::CUDAGuard guard(a.device());
  int64_t n = a.numel();
  if (!n) return;
  const float *ap=a.data_ptr<float>(), *bp=b.data_ptr<float>();
  float *cp=c.data_ptr<float>();
  // Contiguous slices can still have an unaligned storage offset.
  bool aligned = (reinterpret_cast<uintptr_t>(ap)%16==0 && reinterpret_cast<uintptr_t>(bp)%16==0 && reinterpret_cast<uintptr_t>(cp)%16==0);
  bool vec = vectorized && aligned;
  int64_t work = vec ? (n+3)/4 : n;
  int blocks = int(std::min<int64_t>((work+threads-1)/threads, 4096));
  auto stream = at::cuda::getCurrentCUDAStream();
  if(vec) add_vector<<<blocks,threads,0,stream>>>(ap,bp,cp,n);
  else add_scalar<<<blocks,threads,0,stream>>>(ap,bp,cp,n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("add", &add);}
