#include "cholesky.hpp"
#include <omp.h>
#include <cmath>
#include <stdexcept>

void cholesky_decomposition(const std::vector<std::vector<double>>& A,
                            std::vector<std::vector<double>>& L) {
    int n = A.size();

    // Initialize L with zeros
    L.assign(n, std::vector<double>(n, 0.0));

    for (int i = 0; i < n; ++i) {
        // Compute diagonal element L[i][i]
        double sum = 0.0;

        #pragma omp parallel for reduction(+:sum)
        for (int k = 0; k < i; ++k) {
            sum += L[i][k] * L[i][k];
        }

        double diag = A[i][i] - sum;
        if (diag <= 0.0) {
            throw std::runtime_error("Matrix is not positive definite");
        }
        L[i][i] = std::sqrt(diag);

        // Compute off-diagonal elements L[i][j] for j > i
        #pragma omp parallel for
        for (int j = i + 1; j < n; ++j) {
            double sum = 0.0;

            for (int k = 0; k < i; ++k) {
                sum += L[j][k] * L[i][k];
            }

            L[j][i] = (A[j][i] - sum) / L[i][i];
        }
    }
}
