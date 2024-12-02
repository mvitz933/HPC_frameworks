#include <omp.h>
#include "jacobi.hpp" 
#include <iostream>
#include <vector>
#include <cmath>

// Function to initialize the grid, boundary values, and source term
void initialize(int n, int m, float dx, float dy,
                std::vector<std::vector<float>>& u,
                std::vector<std::vector<float>>& unew,
                std::vector<std::vector<float>>& f) {
    const float pi = 2.0f * asinf(1.0f);

    // Set boundary conditions and initialize `u` and `unew`
    for (int i = 0; i < m; i++) {
        u[0][i] = 0.0f;       // Bottom boundary
        unew[0][i] = 0.0f;    // Bottom boundary
        u[n - 1][i] = 0.0f;   // Top boundary
        unew[n - 1][i] = 0.0f; // Top boundary
    }

    for (int j = 0; j < n; j++) {
        float y = j * dy;
        u[j][0] = sinf(pi * y); // Left boundary condition
        unew[j][0] = sinf(pi * y);
        u[j][m - 1] = sinf(pi * y) * expf(-pi); // Right boundary condition
        unew[j][m - 1] = sinf(pi * y) * expf(-pi);
    }

    // Initialize the source term `f`
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            f[j][i] = 0.0f;  // Boundary points contribute nothing to `f`
        }
    }
}

int main(int argc, char** argv) {
    int n = 4096;           // Number of rows
    int m = 4096;           // Number of columns
    int iter_max = 100000;    // Maximum iterations
    const float tol = 1.0e-4f; // Tolerance for convergence

    // Grid spacing
    float dx = 1.0f / (n - 1);
    float dy = 1.0f / (m - 1);

    // Allocate memory using std::vector
    std::vector<std::vector<float>> u(n, std::vector<float>(m, 0.0f));       // Current solution
    std::vector<std::vector<float>> unew(n, std::vector<float>(m, 0.0f));    // Next iteration solution
    std::vector<std::vector<float>> f(n, std::vector<float>(m, 0.0f));       // Source term

    // Initialize the grid, boundary values, and source term
    initialize(n, m, dx, dy, u, unew, f);

    std::cout << "Jacobi relaxation Calculation: " << n << " x " << m << " mesh\n";

    double error=1.0;
    // Call the relaxation function
    jacobi(n, m, iter_max, tol, dx, dy, error, f, u, unew);

    return 0;
}