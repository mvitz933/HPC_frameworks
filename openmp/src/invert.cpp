#include "invert.hpp"
#include <stdexcept>
#include <cmath>
#include <omp.h>

std::vector<std::vector<double>> invert_matrix(const std::vector<std::vector<double>>& A) {
    std::size_t n = A.size();

    // Ensure matrix is square
    if (n == 0 || A[0].size() != n) {
        throw std::invalid_argument("Matrix must be square and non-empty.");
    }

    // Create augmented matrix [A | I]
    std::vector<std::vector<double>> augmented(n, std::vector<double>(2 * n, 0.0));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            augmented[i][j] = A[i][j];
        }
        augmented[i][n + i] = 1.0; // Identity matrix on the right
    }

    // Perform Gauss-Jordan elimination
    for (std::size_t k = 0; k < n; ++k) {
        // Find the pivot element
        double max_val = std::fabs(augmented[k][k]);
        std::size_t pivot_row = k;
        for (std::size_t i = k + 1; i < n; ++i) {
            if (std::fabs(augmented[i][k]) > max_val) {
                max_val = std::fabs(augmented[i][k]);
                pivot_row = i;
            }
        }

        // Check for singularity
        if (max_val < 1e-12) {
            throw std::runtime_error("Matrix is singular or nearly singular.");
        }

        // Swap rows if necessary
        if (pivot_row != k) {
            std::swap(augmented[k], augmented[pivot_row]);
        }

        // Normalize pivot row
        double pivot = augmented[k][k];
        #pragma omp parallel for
        for (std::size_t j = 0; j < 2 * n; ++j) {
            augmented[k][j] /= pivot;
        }

        // Eliminate other rows
        #pragma omp parallel for
        for (std::size_t i = 0; i < n; ++i) {
            if (i != k) {
                double factor = augmented[i][k];
                for (std::size_t j = 0; j < 2 * n; ++j) {
                    augmented[i][j] -= factor * augmented[k][j];
                }
            }
        }
    }

    // Extract the inverse matrix from the augmented matrix
    std::vector<std::vector<double>> inverse(n, std::vector<double>(n, 0.0));
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            inverse[i][j] = augmented[i][n + j];
        }
    }

    return inverse;
}
