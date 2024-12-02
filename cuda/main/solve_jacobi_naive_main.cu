#include <stdio.h>
#include <cuda_runtime.h>
#include "jacobi_solver_naive.hpp"

#define N 16 // Matrix size (N x N)
#define T 100  // Number of Jacobi iterations

void initialize_matrix(float *matrix, int n, float value) {
    for (int i = 0; i < n * n; i++) {
        matrix[i] = value;
    }
}

void print_matrix(const float *matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.2f ", matrix[i * n + j]);
        }
        printf("\n");
    }
}

int main() {
    float *host_input, *host_output;
    float *dev_input, *dev_output;

    // Allocate host memory
    host_input = (float *)malloc(N * N * sizeof(float));
    host_output = (float *)malloc(N * N * sizeof(float));

    // Initialize matrices
    initialize_matrix(host_input, N, 1.0f);
    initialize_matrix(host_output, N, 0.0f);

    // Allocate device memory
    cudaMalloc((void **)&dev_input, N * N * sizeof(float));
    cudaMalloc((void **)&dev_output, N * N * sizeof(float));

    // Copy input matrix to device
    cudaMemcpy(dev_input, host_input, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // Configure kernel launch parameters
    dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 dimGrid((N + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (N + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);

    // Execute Jacobi iterations
    for (int t = 0; t < T; t++) {
        printf("iteration # %.2d ", t);

        naive_jacobi<<<dimGrid, dimBlock>>>(dev_input, dev_output, N);
        cudaDeviceSynchronize();

        // Swap input and output matrices
        float *temp = dev_input;
        dev_input = dev_output;
        dev_output = temp;
    }

    // Copy result back to host
    cudaMemcpy(host_output, dev_input, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the resulting matrix (optional, for small sizes)
    print_matrix(host_output, N);

    // Free memory
    free(host_input);
    free(host_output);
    cudaFree(dev_input);
    cudaFree(dev_output);

    return 0;
}
