#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#define CHECK(x) do { auto e=(x); if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e)); return 1;} }while(0)
__global__ void fill(int *p) { p[threadIdx.x]=threadIdx.x; }
int main(){
 cudaDeviceProp p{}; CHECK(cudaGetDeviceProperties(&p,0));
 printf("GPU=%s SM=%d.%d VRAM=%zu MiB SM_count=%d\n",p.name,p.major,p.minor,p.totalGlobalMem/(1024*1024),p.multiProcessorCount);
 int *d; int h[32]; CHECK(cudaMalloc(&d,sizeof(h)));
 fill<<<1,32>>>(d); CHECK(cudaGetLastError());
 CHECK(cudaMemcpy(h,d,sizeof(h),cudaMemcpyDeviceToHost)); CHECK(cudaFree(d));
 for(int i=0;i<32;i++) if(h[i]!=i) return 2;
 puts("PASS: nvcc, CUDA runtime, kernel launch, GPU copy");
}
