#include "lu.hpp"
#include <vector>
#include <stdexcept>
#include <iostream>
#include <algorithm> // For std::swap
#include <omp.h>

void lu_decomposition(const std::vector<std::vector<double>>& A,
                      std::vector<std::vector<double>>& L,
                      std::vector<std::vector<double>>& U,
                      std::vector<std::vector<double>>& P) {
    std::size_t n = A.size();

    if (n == 0 || A[0].size() != n) {
        throw std::invalid_argument("Matrix must be square and non-empty.");
    }

    // Initialize L, U, and P
    L = std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0));
    U = A; // Start with A as U
    P = std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0));
    for (std::size_t i = 0; i < n; ++i) {
        P[i][i] = 1.0; // Initialize P as the identity matrix
    }

    for (std::size_t k = 0; k < n; ++k) {
        // Find the pivot row (serial operation)
        double max_val = std::abs(U[k][k]);
        std::size_t pivot_row = k;
        for (std::size_t i = k + 1; i < n; ++i) {
            if (std::abs(U[i][k]) > max_val) {
                max_val = std::abs(U[i][k]);
                pivot_row = i;
            }
        }

        // Swap rows in U and P
        if (pivot_row != k) {
            std::swap(U[k], U[pivot_row]);
            std::swap(P[k], P[pivot_row]);
            if (k > 0) {
                std::swap(L[k], L[pivot_row]);
            }
        }

        // Compute the lower triangular matrix L and update U
        #pragma omp parallel for
        for (std::size_t i = k + 1; i < n; ++i) {
            double factor = U[i][k] / U[k][k];
            L[i][k] = factor;

            #pragma omp parallel for
            for (std::size_t j = k; j < n; ++j) {
                U[i][j] -= factor * U[k][j];
            }
        }
        L[k][k] = 1.0; // Diagonal of L is 1
    }
}
