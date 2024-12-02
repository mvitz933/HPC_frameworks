/* Sequential Solution to Steady-State Heat Problem */
#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <fstream>
#include <sstream>
#include <omp.h>
#include "jacobi.hpp" 

void initialize(int n, int m, float& mean, float topBC, float bottomBC, float edgeBC, 
                std::vector<std::vector<float>>& u, std::vector<std::vector<float>>& f) {
    // Set boundary values and compute mean boundary value
    mean = 0.0f;

    #pragma omp parallel for reduction(+:mean)
    for (int i = 1; i < n + 1; ++i) {
        u[0][i] = bottomBC;  // Bottom boundary
        u[n + 1][i] = topBC; // Top boundary
        u[i][0] = edgeBC;    // Left boundary
        u[i][m + 1] = edgeBC; // Right boundary

        mean += u[0][i] + u[n + 1][i] + u[i][0] + u[i][m + 1];
    }
    mean /= (2.0f * (n + m));

    // Initialize interior values and f
    #pragma omp parallel for collapse(2)
    for (int j = 1; j <= n; ++j) {
        for (int i = 1; i <= m; ++i) {
            u[j][i] = mean;
            f[j][i] = 0.0f; // No external heat sources for the Laplace problem
        }
    }
}
int main(int argc, char* argv[]) {
    const int n = 500;               // Grid size in the x-direction
    const int m = 500;               // Grid size in the y-direction
    const int iter_max = 1000000;    // Maximum iterations
    const float tol = 1.0e-4f;       // Convergence tolerance
    const float dx = 1.0f / n;       // Grid spacing in x-direction
    const float dy = 1.0f / m;       // Grid spacing in y-direction

    // Boundary conditions
    const float topBC = 100.0f;
    const float bottomBC = 0.0f;
    const float edgeBC = 100.0f;

    // Allocate memory for the grids
    std::vector<std::vector<float>> u(n + 2, std::vector<float>(m + 2, 0.0f));
    std::vector<std::vector<float>> unew(n + 2, std::vector<float>(m + 2, 0.0f));
    std::vector<std::vector<float>> f(n + 2, std::vector<float>(m + 2, 0.0f));

    // Initialize the grid and boundary values
    float mean = 0.0f;
    initialize(n, m, mean, topBC, bottomBC, edgeBC, u, f);

    std::cout << "Jacobi relaxation calculation: " << n << " x " << m << " grid\n";

    double error=1.0;
    // Call the relaxation function
    jacobi(n, m, iter_max, tol, dx, dy, error, f, u, unew);

    // Write results to a file
    std::ofstream outFile("output.txt");
    if (outFile.is_open()) {
        for (int j = 1; j <= n; ++j) {
            for (int i = 1; i <= m; ++i) {
                outFile << u[j][i] << " ";
            }
            outFile << "\n";
        }
        outFile.close();
        std::cout << "Results written to output.txt\n";
    } else {
        std::cerr << "Error: Could not open output file.\n";
    }

    return 0;
}