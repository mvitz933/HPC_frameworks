#include "qr.hpp"
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <omp.h>

void qr_decomposition(const std::vector<std::vector<double>>& A,
                      std::vector<std::vector<double>>& Q,
                      std::vector<std::vector<double>>& R) {
    std::size_t m = A.size();       // Number of rows
    std::size_t n = A[0].size();    // Number of columns

    if (m < n) {
        throw std::invalid_argument("Matrix A must have at least as many rows as columns.");
    }

    // Initialize Q and R matrices
    Q = std::vector<std::vector<double>>(m, std::vector<double>(n, 0.0));
    R = std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0));

    std::vector<std::vector<double>> U = A;  // Copy A to U for processing

    for (std::size_t k = 0; k < n; ++k) {
        // Compute R[k][k] as the norm of the k-th column of U
        double norm = 0.0;
        #pragma omp parallel for reduction(+:norm)
        for (std::size_t i = 0; i < m; ++i) {
            norm += U[i][k] * U[i][k];
        }
        R[k][k] = std::sqrt(norm);

        if (R[k][k] == 0) {
            throw std::runtime_error("Matrix is rank deficient.");
        }

        // Compute the k-th column of Q
        #pragma omp parallel for
        for (std::size_t i = 0; i < m; ++i) {
            Q[i][k] = U[i][k] / R[k][k];
        }

        // Update the remaining columns of U and compute R[k][j] for j > k
        for (std::size_t j = k + 1; j < n; ++j) {
            double dot_product = 0.0;
            #pragma omp parallel for reduction(+:dot_product)
            for (std::size_t i = 0; i < m; ++i) {
                dot_product += Q[i][k] * U[i][j];
            }
            R[k][j] = dot_product;

            #pragma omp parallel for
            for (std::size_t i = 0; i < m; ++i) {
                U[i][j] -= R[k][j] * Q[i][k];
            }
        }
    }
}
