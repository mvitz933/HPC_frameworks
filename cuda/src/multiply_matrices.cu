#include <stdio.h>
#include <cuda_runtime.h>
#include "multiply_matrices.hpp"

// Template-based CUDA kernel with restrict keyword
__global__ void multiply_matrices(const double A[ROWS_A][COLS_A], 
                                  const double B[COLS_A][COLS_B], 
                                  double C[ROWS_A][COLS_B]){
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // Row index of C
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // Column index of C

    if (row < ROWS_A && col < COLS_B) {
        double sum = 0.0;
        for (int k = 0; k < COLS_A; ++k) {
            sum += A[row][k] * B[k][col];
        }
        C[row][col] = sum;
    }
}

