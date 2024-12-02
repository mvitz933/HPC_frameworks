#include <stdio.h>
#include <cuda_runtime.h>

constexpr int N = 4;  // Square matrix size

// Matrix declared with __managed__
__managed__ double A[N][N];
__managed__ double Inv[N][N];


__global__ void gauss_jordan(double A[N][N], double Inv[N][N], int N, int i) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // Row
    int y = blockIdx.y * blockDim.y + threadIdx.y; // Column

    if (x < N && y < N) {
        if (x > i) { // Only process rows below the pivot
            double P = A[x][i] / A[i][i];

            // Update the inverse matrix
            Inv[x][y] -= Inv[i][y] * P;

            // Update the A matrix (only to the right of the pivot)
            if (y >= i) {
                A[x][y] -= A[i][y] * P;
            }
        }
    }
}

__global__ void dev(double A[N][N], double Inv[N][N], int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // Row index
    int y = blockIdx.y * blockDim.y + threadIdx.y; // Column index

    if (x < h && y < h) {
        // Check if the pivot element (diagonal) is non-zero
        if (A[x][x] != 0) {
            // Normalize the corresponding row in both A and Inv
            Inv[x][y] /= A[x][x];
            A[x][y] /= A[x][x];
        }
    }

    __syncthreads(); // Ensure all threads synchronize after row normalization
}




// CUDA kernel for matrix inversion using Gauss-Jordan elimination
__global__ void invert_matrix_old(double A[N][N], double Inv[N][N]) {
    int row =  threadIdx.x;  // Row index
    int col =  threadIdx.y;  // Column index

    for (int k = 0; k < N; ++k) {
        if (row == k) {
            double pivot = A[row][k];
            if (fabs(pivot) < 1e-12) {
                if (row == 0) printf("Matrix is singular or nearly singular!\n");
                return;  // Exit if pivot is invalid
            }
            for (col = 0; col < N; ++col) {
                A[row][col] /= pivot;  // Normalize A
                Inv[row][col] /= pivot;  // Normalize Inv
            }
        }
        __syncthreads();

        // Eliminate other rows
        // Step 2: Eliminate the current column for all other rows
        if (row != k) {
            double factor = A[row][k];
            for (col = 0; col < N; ++col) {
                A[row][col] -= factor * A[k][col];  // Update A
                Inv[row][col] -= factor * Inv[k][col];  // Update Inv
            }
        }
        __syncthreads();
    }
}


int main() {
    // Initialize matrix A
    printf("Input matrix A:\n");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A[i][j] = (i == j) ? 2.0 : (i + j + 1);  // Diagonal dominance for invertibility
            Inv[i][j] = (i == j) ? 1.0 : 0.0;       // Initialize Inv to the identity matrix
            printf("%f ", A[i][j]);
        }
        printf("\n");
    }

    // Configure kernel launch parameters
    dim3 threadsPerBlock(N, 1);
    dim3 blocksPerGrid(1, 1);
    dim3 numBlocks(16,16);

    // Launch kernel
    //invert_matrix<<<blocksPerGrid, threadsPerBlock>>>(A, Inv);

    for(int i=0;i<N;i++){
        gauss_jordan<<<numBlocks,threadsPerBlock>>>(A, Inv, N, i);
        cudaDeviceSynchronize();
    }
    //dev<<<numBlocks,threadsPerBlock>>>(A, Inv, N);


    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Print the inverted matrix
    printf("\nInverted matrix Inv:\n");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%f ", Inv[i][j]);
        }
        printf("\n");
    }

    return 0;
}
