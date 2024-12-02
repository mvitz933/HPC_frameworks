
#include "multiply_matrices.hpp"
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cuda_runtime.h>



int main() {
    //initializing the matrices
    for (int i = 0; i < ROWS_A; ++i) {
        for (int j = 0; j < COLS_A; ++j) {
            A[i][j] = 1.0 * (i + j + 1)/(COLS_A*COLS_B);  // Fill A with sequential values
        }
    }
    for (int i = 0; i < COLS_A; ++i) {
        for (int j = 0; j < COLS_B; ++j) {
            B[i][j] = 1.0 * (i + j + 1)/(COLS_A*COLS_B);  // Fill B with sequential values
        }
    }
    C[ROWS_A][COLS_B] = {0};

    // Configure kernel launch parameters
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((COLS_B + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ROWS_A + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch kernel
    multiply_matrices<<<blocksPerGrid, threadsPerBlock>>>(A, B, C);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Print result matrix
    printf("Result matrix C:\n");
    for (int i = 0; i < ROWS_A; ++i) {
        for (int j = 0; j < COLS_B; ++j) {
            printf("%f ", C[i][j]);
        }
        printf("\n");
    }

    return 0;
}

