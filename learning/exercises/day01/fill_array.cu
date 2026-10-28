#include <cstdio>
#include <cuda_runtime.h>



//__global__修饰符表示：在host调用，在device上运行
__global__ void fill_array(int *arr, int value, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        arr[idx] = value;
    }
}

int main(){
    const int N = 32;
    const int VALUE = 42;

    //host端
    int h_arr[N];
    int *d_arr = nullptr;

    cudaMalloc(&d_arr, N * sizeof(int));
    //启动kernel：1个block，32个线程
    fill_array<<<1, 32>>>(d_arr, VALUE, N);

    cudaDeviceSynchronize();
    cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);

    for(int i = 0; i<N ; i++){
        printf("%d ", h_arr[i]);
    }
    printf("\n");
    cudaFree(d_arr);
    return 0;
}