#include "transpose_contract.cuh"
// 输入 [rows,cols] -> 输出 [cols,rows]，row-major。
__global__ void naive_read(const float* a,float* b,int64_t rows,int64_t cols) {
    int64_t x=int64_t(blockIdx.x)*32+threadIdx.x;
    int64_t y=int64_t(blockIdx.y)*8+threadIdx.y;
    if(x<cols&&y<rows)b[x*rows+y]=a[y*cols+x]; // 读合并，写跨行。
}
__global__ void naive_write(const float* a,float* b,int64_t rows,int64_t cols) {
    int64_t x=int64_t(blockIdx.x)*32+threadIdx.x;
    int64_t y=int64_t(blockIdx.y)*8+threadIdx.y;
    if(x<rows&&y<cols)b[y*rows+x]=a[x*cols+y]; // 写合并，读跨行。
}
__global__ void transpose_tiled(const float* a,float* b,int64_t rows,int64_t cols) {
    __shared__ float tile[32][32];
    int tx=threadIdx.x,ty=threadIdx.y;
    int64_t x=int64_t(blockIdx.x)*32+tx,y=int64_t(blockIdx.y)*32+ty;
#pragma unroll
    for(int j=0;j<32;j+=8)tile[ty+j][tx]=(x<cols&&y+j<rows)?a[(y+j)*cols+x]:0;
    __syncthreads(); // 所有线程参与，不能放进越界条件里。
    x=int64_t(blockIdx.y)*32+tx;y=int64_t(blockIdx.x)*32+ty;
#pragma unroll
    for(int j=0;j<32;j+=8)if(x<rows&&y+j<cols)b[(y+j)*rows+x]=tile[tx][ty+j];
}
__global__ void copy_flat(const float* a,float* b,int64_t n) {
    for(int64_t i=int64_t(blockIdx.x)*256+threadIdx.x;i<n;i+=int64_t(gridDim.x)*256)b[i]=a[i];
}
void transpose(torch::Tensor a,torch::Tensor b,int version) {
    check_transpose(a,b);TORCH_CHECK(version>=0&&version<=2,"version must be 0/1/2");
    const c10::cuda::CUDAGuard guard(a.device());if(!a.numel())return;
    int64_t rows=a.size(0),cols=a.size(1);auto stream=at::cuda::getCurrentCUDAStream();dim3 threads(32,8);
    if(version==0)naive_read<<<dim3((cols+31)/32,(rows+7)/8),threads,0,stream>>>(a.data_ptr<float>(),b.data_ptr<float>(),rows,cols);
    if(version==1)naive_write<<<dim3((rows+31)/32,(cols+7)/8),threads,0,stream>>>(a.data_ptr<float>(),b.data_ptr<float>(),rows,cols);
    if(version==2)transpose_tiled<<<dim3((cols+31)/32,(rows+31)/32),threads,0,stream>>>(a.data_ptr<float>(),b.data_ptr<float>(),rows,cols);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void copy(torch::Tensor a,torch::Tensor b) {
    check_transpose(a,b);const c10::cuda::CUDAGuard guard(a.device());if(!a.numel())return;
    copy_flat<<<int(std::min<int64_t>((a.numel()+255)/256,4096)),256,0,at::cuda::getCurrentCUDAStream()>>>(a.data_ptr<float>(),b.data_ptr<float>(),a.numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("transpose",&transpose);m.def("copy",&copy);}
