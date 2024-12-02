#include "eigenvalues.hpp"
#include "qr.hpp"
#include <vector>
#include <cmath>
#include <omp.h>
#include <stdexcept>
#include <iostream>


std::vector<double> calculate_all_eigenvalues(const std::vector<std::vector<double>>& A, int max_iters, double tol) {
    std::size_t n = A.size();
    if (n == 0 || A[0].size() != n) {
        throw std::invalid_argument("Matrix must be square and non-empty.");
    }

    // Copy matrix A for iteration
    std::vector<std::vector<double>> B = A;

    for (int iter = 0; iter < max_iters; ++iter) {
        std::vector<std::vector<double>> Q, R;

        // Perform QR decomposition
        qr_decomposition(B, Q, R);

        // Compute B = R * Q
        std::vector<std::vector<double>> B_next(n, std::vector<double>(n, 0.0));
        #pragma omp parallel for collapse(2)
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                for (std::size_t k = 0; k < n; ++k) {
                    B_next[i][j] += R[i][k] * Q[k][j];
                }
            }
        }

        // Check for convergence (max off-diagonal element)
        double max_off_diag = 0.0;
        #pragma omp parallel for reduction(max:max_off_diag)
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                if (i != j) {
                    max_off_diag = std::max(max_off_diag, std::abs(B_next[i][j]));
                }
            }
        }

        B = std::move(B_next);

        if (max_off_diag < tol) break;
    }

    // Extract eigenvalues from the diagonal of B
    std::vector<double> eigenvalues(n);
    #pragma omp parallel for
    for (std::size_t i = 0; i < n; ++i) {
        eigenvalues[i] = B[i][i];
    }

    return eigenvalues;
}
