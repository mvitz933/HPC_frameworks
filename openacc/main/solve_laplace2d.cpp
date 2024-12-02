#include <omp.h>
#include "jacobi.hpp" 
#include <iostream>
#include <vector>
#include <cmath>

// Function to initialize the grid, boundary values, and source term
template <int N, int M>
void initialize(float dx, float dy, 
                double u[N][N], double unew[N][N], double f[N][N]) {
    const float pi = 2.0f * asinf(1.0f);

    // Set boundary conditions and initialize `u` and `unew`
    for (int i = 0; i < M; i++) {
        u[0][i] = 0.0;       // Bottom boundary
        unew[0][i] = 0.0;    // Bottom boundary
        u[N - 1][i] = 0.0;   // Top boundary
        unew[N - 1][i] = 0.0; // Top boundary
    }

    for (int j = 0; j < N; j++) {
        float y = j * dy;
        u[j][0] = sin(pi * y); // Left boundary condition
        unew[j][0] = sin(pi * y);
        u[j][M - 1] = sin(pi * y) * exp(-pi); // Right boundary condition
        unew[j][M - 1] = sin(pi * y) * exp(-pi);
    }

    // Initialize the source term `f`
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < M; i++) {
            f[j][i] = 0.0f;  // Boundary points contribute nothing to `f`
        }
    }
}


int main() {
    const int N = 512;      // Number of rows
    const int M = 512;      // Number of columns
    const int iter_max = 1001;    // Maximum iterations
    const float tol = 1.0e-4f; // Tolerance for convergence
    std::cout << "set the constants.\n";
    
    // Grid spacing
    const float dx = 1.0f / (N - 1);
    const float dy = 1.0f / (M - 1);
    
    double f[N][N] = {0.0};    // Source term
    double u[N][N] = {0.0};    // Current values
    double unew[N][N] = {0.0}; // Updated values

    std::cout << "Initializing matrices.\n";

    // Initialize the grid, boundary values, and source term
    initialize<N,M>(dx, dy, u, unew, f);
    std::cout << "Beginning Jacobi relaxation Calculation: " << N << " x " << M << " mesh\n";

    double error=1.0;
    // Call the relaxation function
    jacobi<N,M>(iter_max, tol, dx, dy, error, f, u, unew);

    return 0;
}
