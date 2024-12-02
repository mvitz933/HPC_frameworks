#pragma once

#include <cmath>
#include <stdexcept>
#include <openacc.h>
#include "qr.hpp"

// Function to calculate all eigenvalues of a square matrix using the QR algorithm
// Input: A (square matrix), max_iters (maximum number of iterations), tol (tolerance for convergence)
// Output: A vector of eigenvalues
template <int N>
void calculate_all_eigenvalues(const double A[N][N], double eigenvalues[N], int max_iters, double tol) {
    // Copy matrix A into B for iteration
    double B[N][N];

    // Initialize B with A
    #pragma acc parallel loop collapse(2)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            B[i][j] = A[i][j];
        }
    }

    for (int iter = 0; iter < max_iters; ++iter) {
        double Q[N][N]; // No need to initialize to zero; will be set in qr_decomposition
        double R[N][N];

        // Perform QR decomposition using your function
        qr_decomposition<N, N>(B, Q, R);

        // Compute B_next = R * Q
        double B_next[N][N];

        #pragma acc parallel loop collapse(2)
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                double sum = 0.0;
                for (int k = 0; k < N; ++k) {
                    sum += R[i][k] * Q[k][j];
                }
                B_next[i][j] = sum;
            }
        }

        // Check for convergence
        double diff = 0.0;

        #pragma acc parallel loop collapse(2) reduction(+:diff)
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                diff += fabs(B[i][j] - B_next[i][j]);
            }
        }

        // Update B
        #pragma acc parallel loop collapse(2)
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                B[i][j] = B_next[i][j];
            }
        }

        // Check for convergence
        if (diff < tol) {
            break;
        }
    }

    // Extract eigenvalues from the diagonal of B
    #pragma acc parallel loop
    for (int i = 0; i < N; ++i) {
        eigenvalues[i] = B[i][i];
    }
}
