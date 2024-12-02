#include "add_vectors.hpp"
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CALL(x) do { \
    cudaError_t err = x; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)



int main() {

    // Initialize managed memory arrays
    for (int i = 0; i < N; i++) {
        vector_a[i] = 1.0;
        vector_b[i] = 2.0;
    }


    // Configure kernel launch parameters
    int thr_per_blk = 256;
    int blk_in_grid = (N + thr_per_blk - 1) / thr_per_blk;

    // Launch kernel
    add_vectors<<<blk_in_grid, thr_per_blk>>>(vector_a, vector_b, vector_c, N);

    CUDA_CALL(cudaDeviceSynchronize());

    // Verify results
    double tolerance = 1.0e-14;
    for (int i = 0; i < N; i++) {
        if (fabs(vector_c[i] - 3.0) > tolerance) {
            printf("\nError: value of vector_c[%d] = %f instead of 3.0\n", i, vector_c[i]);
            return 1;
        }
    }

    printf("\n---------------------------\n");
    printf("__SUCCESS__\n");
    printf("---------------------------\n");
    printf("N                 = %d\n", N);
    printf("Threads Per Block = %d\n", thr_per_blk);
    printf("Blocks In Grid    = %d\n", blk_in_grid);
    printf("---------------------------\n\n");

    return 0;
}
