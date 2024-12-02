#pragma once

#include <vector>
#include <stdexcept>
#include "lu.hpp"
// Solve Ax = b using LU decomposition
#include <cmath>
#include <stdexcept>
#include <openacc.h>

// Function to solve Ax = b using LU decomposition
template <int N>
void solve_linear_system_with_lu(const double A[N][N], const double b[N], double x[N]) {
    // Variables for LU decomposition
    double L[N][N], U[N][N], P[N][N];

    // Perform LU decomposition (assuming the function is GPU-compatible)
    lu_decomposition<N>(A, L, U, P);

    // Apply permutation matrix P to vector b to get Pb
    double Pb[N];
    #pragma acc parallel loop
    for (int i = 0; i < N; ++i) {
        Pb[i] = 0.0;
        for (int j = 0; j < N; ++j) {
            Pb[i] += P[i][j] * b[j];
        }
    }

    // Forward substitution: Solve Ly = Pb
    double y[N];
    double sum = 0.0;
    for (int i = 0; i < N; ++i) {
        sum = 0.0;
        #pragma acc parallel loop reduction(+:sum)
        for (int j = 0; j < i; ++j) {
            sum += L[i][j] * y[j];
        }
        y[i] = Pb[i] - sum;
    }

    // Back substitution: Solve Ux = y
    sum = 0.0;
    for (int i = N - 1; i >= 0; --i) {
        sum = 0.0;
        #pragma acc parallel loop reduction(+:sum)
        for (int j = i + 1; j < N; ++j) {
            sum += U[i][j] * x[j];
        }
        x[i] = (y[i] - sum) / U[i][i];
    }
}
