#pragma once

#include <iostream>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <openacc.h>

// QR Decomposition Function with Templates
template <int M, int N>
void qr_decomposition(const double A[M][N], double Q[M][N], double R[N][N]) {
    // Ensure the input matrix dimensions are valid
    static_assert(M >= N, "Matrix A must have at least as many rows as columns.");

    // Create a copy of A for modification
    double U[M][N];

    // Initialize U with values from A
    #pragma acc parallel loop collapse(2)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            U[i][j] = A[i][j];
        }
    }

    // QR decomposition logic
    for (int k = 0; k < N; ++k) {
        // Compute R[k][k] as the norm of the k-th column of U
        double norm = 0.0;
        #pragma acc parallel loop reduction(+:norm)
        for (int i = 0; i < M; ++i) {
            norm += U[i][k] * U[i][k];
        }

        // Compute sqrt on the device
        #pragma acc parallel
        {
            norm = sqrt(norm);
            R[k][k] = norm;
        }

        // Handle rank deficiency (error checking on device)
        if (norm == 0.0) {
            R[k][k] = 1e-10; // Avoid division by zero
        }

        // Compute the k-th column of Q
        #pragma acc parallel loop
        for (int i = 0; i < M; ++i) {
            Q[i][k] = U[i][k] / R[k][k];
        }

        // Update the remaining columns of U and compute R[k][j] for j > k
        for (int j = k + 1; j < N; ++j) {
            double dot_product = 0.0;
            #pragma acc parallel loop reduction(+:dot_product)
            for (int i = 0; i < M; ++i) {
                dot_product += Q[i][k] * U[i][j];
            }

            // Assign R[k][j] on the device
            #pragma acc parallel
            {
                R[k][j] = dot_product;
            }

            // Update U[i][j] on the device
            #pragma acc parallel loop
            for (int i = 0; i < M; ++i) {
                U[i][j] -= R[k][j] * Q[i][k];
            }
        }
    }
}
