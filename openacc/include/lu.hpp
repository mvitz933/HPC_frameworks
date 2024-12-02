#pragma once

#include <vector>

// Perform LU decomposition on a square matrix A
//void lu_decomposition(const double* A, double* L, double* U, double* P, int n);

#include <cmath>
#include <stdexcept>
#include <iostream>
#include <vector>
#include <openacc.h>


#pragma acc routine
template <typename T>
inline void gpu_swap(T& a, T& b) {
    T temp = a;
    a = b;
    b = temp;
}

// LU Decomposition with OpenACC and Pivoting using Templates and GPU-friendly code
template <int N>
void lu_decomposition(const double A[N][N], double L[N][N], double U[N][N], double P[N][N]) {
    // Variables for error checking

    bool is_error = false;
    int error_index = -1;

    // Initialize matrice
    #pragma acc parallel loop collapse(2)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            L[i][j] = 0.0;
            U[i][j] = A[i][j];
            P[i][j] = (i == j) ? 1.0 : 0.0; // Identity matrix for P
        }
    }

    // Perform LU decomposition with pivoting
    for (int k = 0; k < N; ++k) {
        // Step 1: Find pivot row (serial operation)
        double max_val = fabs(U[k][k]); // Using fabs instead of std::abs
        int pivot_row = k;
        for (int i = k + 1; i < N; ++i) {
            double abs_val = fabs(U[i][k]);
            if (abs_val > max_val) {
                max_val = abs_val;
                pivot_row = i;
            }
        }

        // Step 2: Swap rows in U, P, and L (if k > 0)
        if (pivot_row != k) {
            #pragma acc parallel loop
            for (int j = 0; j < N; ++j) {
                // Swap U[k][j] and U[pivot_row][j]
                double temp = U[k][j];
                U[k][j] = U[pivot_row][j];
                U[pivot_row][j] = temp;

                // Swap P[k][j] and P[pivot_row][j]
                temp = P[k][j];
                P[k][j] = P[pivot_row][j];
                P[pivot_row][j] = temp;

                if (k > 0) {
                    // Swap L[k][j] and L[pivot_row][j]
                    temp = L[k][j];
                    L[k][j] = L[pivot_row][j];
                    L[pivot_row][j] = temp;
                }
            }
        }

        // Step 3: Check for singular matrix
        if (U[k][k] == 0.0) {
            is_error = true;
            error_index = k * N + k;
            break;
        }

        // Step 4: Compute L and update U
        // Compute the lower triangular matrix L
        #pragma acc parallel loop
        for (int i = k + 1; i < N; ++i) {
            double factor = U[i][k] / U[k][k];
            L[i][k] = factor;

            // Update U matrix
            #pragma acc loop
            for (int j = k; j < N; ++j) {
                U[i][j] -= factor * U[k][j];
            }
        }

        // Set diagonal of L to 1
        L[k][k] = 1.0;

        // Step 5: Check for NaN or errors
        is_error = false; // Reset is_error for reduction
        #pragma acc parallel loop collapse(2) reduction(||:is_error)
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                // Replace std::isnan with x != x
                if (U[i][j] != U[i][j] || L[i][j] != L[i][j]) {
                    is_error = true;
                    error_index = i * N + j;
                }
            }
        }

    }

    // Handle errors
    if (is_error) {
        throw std::runtime_error("LU decomposition failed. Matrix may not be positive definite or is singular at index: " + std::to_string(error_index));
    }
}
