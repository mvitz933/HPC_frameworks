#pragma once

#include <cmath>
#include <stdexcept>
#include <openacc.h>

// Function to invert a square matrix using Gauss-Jordan elimination
template <int N>
void invert_matrix(const double A[N][N], double inverse[N][N]) {
    // Ensure the input matrix is square
    static_assert(N > 0, "Matrix size must be greater than 0.");

    // Create an augmented matrix [A | I]
    double augmented[N][2 * N];

    // Initialize the augmented matrix
    #pragma acc parallel loop collapse(2)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < 2 * N; ++j) {
            if (j < N) {
                augmented[i][j] = A[i][j];
            } else {
                augmented[i][j] = (i == (j - N)) ? 1.0 : 0.0; // Identity matrix
            }
        }
    }

    // Perform Gauss-Jordan elimination
    for (int k = 0; k < N; ++k) {
        // Find the pivot element (serial operation)
        double max_val = fabs(augmented[k][k]);
        int pivot_row = k;

        for (int i = k + 1; i < N; ++i) {
            double abs_val = fabs(augmented[i][k]);
            if (abs_val > max_val) {
                max_val = abs_val;
                pivot_row = i;
            }
        }

        // Check for singularity
        if (max_val < 1e-12) {
            throw std::runtime_error("Matrix is singular or nearly singular.");
        }

        // Swap rows if necessary
        if (pivot_row != k) {
            #pragma acc parallel loop
            for (int j = 0; j < 2 * N; ++j) {
                double temp = augmented[k][j];
                augmented[k][j] = augmented[pivot_row][j];
                augmented[pivot_row][j] = temp;
            }
        }

        // Normalize pivot row
        double pivot = augmented[k][k];
        #pragma acc parallel loop
        for (int j = 0; j < 2 * N; ++j) {
            augmented[k][j] /= pivot;
        }

        // Eliminate other rows
        #pragma acc parallel loop
        for (int i = 0; i < N; ++i) {
            if (i != k) {
                double factor = augmented[i][k];
                #pragma acc loop
                for (int j = 0; j < 2 * N; ++j) {
                    augmented[i][j] -= factor * augmented[k][j];
                }
            }
        }
    }

    // Extract the inverse matrix from the augmented matrix
    #pragma acc parallel loop collapse(2)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            inverse[i][j] = augmented[i][N + j];
        }
    }
}
