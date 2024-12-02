//compile with 
//nvcc -o cude_runtime cuda_runtime.cpp
#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    int device;
    cudaDeviceProp prop;

    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    printf("CUDA Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Total Global Memory: %zu MB\n", prop.totalGlobalMem / (1024 * 1024));
    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Max Threads Dim: [%d, %d, %d]\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("Max Grid Size: [%d, %d, %d]\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);

    return 0;
}

