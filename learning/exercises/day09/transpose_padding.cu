#include "../day08/transpose_contract.cuh"
// 唯一实验变量是行跨度：32 -> 33；线程布局/加载/存储/同步相同。
template<int PAD>
__global__ void transpose_padding(const float* a,float* b,int64_t rows,int64_t cols) {
    __shared__ float tile[32][32+PAD];
    int tx=threadIdx.x,ty=threadIdx.y;
    int64_t x=int64_t(blockIdx.x)*32+tx,y=int64_t(blockIdx.y)*32+ty;
#pragma unroll
    for(int j=0;j<32;j+=8)tile[ty+j][tx]=(x<cols&&y+j<rows)?a[(y+j)*cols+x]:0;
    __syncthreads();
    x=int64_t(blockIdx.y)*32+tx;y=int64_t(blockIdx.x)*32+ty;
#pragma unroll
    for(int j=0;j<32;j+=8)if(x<rows&&y+j<cols)b[(y+j)*rows+x]=tile[tx][ty+j];
}
void transpose(torch::Tensor a,torch::Tensor b,int version) {
    check_transpose(a,b);TORCH_CHECK(version==0||version==1,"version must be 0/1");
    const c10::cuda::CUDAGuard guard(a.device());if(!a.numel())return;
    auto stream=at::cuda::getCurrentCUDAStream();dim3 threads(32,8),grid((a.size(1)+31)/32,(a.size(0)+31)/32);
    if(version==0)transpose_padding<0><<<grid,threads,0,stream>>>(a.data_ptr<float>(),b.data_ptr<float>(),a.size(0),a.size(1));
    else transpose_padding<1><<<grid,threads,0,stream>>>(a.data_ptr<float>(),b.data_ptr<float>(),a.size(0),a.size(1));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("transpose",&transpose);}
