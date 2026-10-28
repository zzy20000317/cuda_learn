#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/types.h>
#include <cuda_runtime.h>
#include <cstdint>

//十六字节对齐检查
inline bool aligned16(const void* p){
    return (reinterpret_cast<std::uintptr_t>(p) & 0xF) == 0;
}

//标量kernel
__global__ void add_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }   
}

//float 4 kernel
__global__ void add_vec4_kernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c,
                                int n) {
                int n4 = n/4;
                int tid =blockIdx.x * blockDim.x + threadIdx.x;
                if(tid < n4){
                    float4 a_val = reinterpret_cast<const float4*>(a)[tid];
                    float4 b_val = reinterpret_cast<const float4*>(b)[tid];
                    float4 c_val = make_float4(a_val.x + b_val.x, a_val.y + b_val.y, a_val.z + b_val.z, a_val.w + b_val.w);
                    reinterpret_cast<float4*>(c)[tid] = c_val;
 
                    }
                //尾部用grid里前tail个线程处理剩下的元素
                int tail_start = n4 * 4;
                int tail = n - tail_start;
                if(tid < tail){
                    c[tail_start + tid] = a[tail_start + tid] + b[tail_start + tid];
                }
            }

//host
void add(torch::Tensor a, torch::Tensor b, torch::Tensor c,
         int block, bool vec) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda() && c.is_cuda(), "must be CUDA");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && c.is_contiguous(),
                "must be contiguous");
    TORCH_CHECK(a.scalar_type() == at::kFloat, "must be float32");

    int n = a.numel();
    if (n == 0) return;

    auto stream = at::cuda::getCurrentCUDAStream();
    const float* pa = a.data_ptr<float>();
    const float* pb = b.data_ptr<float>();
    float*       pc = c.data_ptr<float>();

    // 只有：用户要向量化 + n>=4 + 三个指针都 16 字节对齐，才走 float4
    bool can_vec = vec && (n >= 4)
                && aligned16(pa) && aligned16(pb) && aligned16(pc);

    if (can_vec) {
        int n4 = n / 4;
        int blocks = (n4 + block - 1) / block;
        // 注意：尾部在 kernel 内用额外线程处理，所以 blocks 要够放下 n4 和 tail
        // 实际上 tail <= 3 < n4 (n>=4)，所以 blocks 按 n4 算就够
        add_vec4_kernel<<<blocks, block, 0, stream>>>(pa, pb, pc, n);
    } else {
        int blocks = (n + block - 1) / block;
        add_kernel<<<blocks, block, 0, stream>>>(pa, pb, pc, n);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &add, "add(a,b,c,block,vec)");
}