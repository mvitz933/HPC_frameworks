#include "solve.hpp"
#include "lu.hpp"
#include <stdexcept>
#include <cmath>
#include <omp.h>


// Solve Ax = b using LU decomposition
std::vector<double> solve_linear_system_with_lu(const std::vector<std::vector<double>>& A,
                                                const std::vector<double>& b) {
    std::size_t n = A.size();

    // Perform LU decomposition
    std::vector<std::vector<double>> L, U, P;
    lu_decomposition(A, L, U, P);

    // Apply permutation matrix P to b
    std::vector<double> b_permuted(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        b_permuted[i] = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            b_permuted[i] += P[i][j] * b[j];
        }
    }

    // Forward substitution: Solve Ly = b_permuted
    std::vector<double> y(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        double sum = 0.0;
        for (std::size_t j = 0; j < i; ++j) {
            sum += L[i][j] * y[j];
        }
        y[i] = (b_permuted[i] - sum); // L[i][i] should be 1
    }

    // Back substitution: Solve Ux = y
    std::vector<double> x(n, 0.0);
    for (int i = n - 1; i >= 0; --i) {
        double sum = 0.0;
        for (std::size_t j = i + 1; j < n; ++j) {
            sum += U[i][j] * x[j];
        }
        if (U[i][i] == 0.0) {
            throw std::runtime_error("Singular matrix detected in U.");
        }
        x[i] = (y[i] - sum) / U[i][i];
    }

    return x;
}
