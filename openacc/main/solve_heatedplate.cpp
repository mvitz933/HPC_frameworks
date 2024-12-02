#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <fstream>
#include <sstream>
#include <omp.h>
#include "jacobi.hpp"

// Function to initialize the grid, boundary values, and source term
template <int N, int M>
void initialize(float dx, float dy, double& mean, float topBC, float bottomBC, float edgeBC, 
                double u[N][M], double f[N][M]) {
    const float pi = 2.0f * asinf(1.0f);

    // Set boundary values and compute mean boundary value
    mean = 0.0;
    #pragma omp parallel for reduction(+:mean)
    for (int i = 0; i < M; ++i) {
        u[0][i] = bottomBC;  // Bottom boundary
        u[N - 1][i] = topBC; // Top boundary
        u[i][0] = edgeBC;    // Left boundary
        u[i][M - 1] = edgeBC; // Right boundary

        mean += u[0][i] + u[N - 1][i] + u[i][0] + u[i][M - 1];
    }
    mean /= (2.0f * (N + M));

    // Initialize interior values and f
    #pragma omp parallel for collapse(2)
    for (int j = 1; j < N - 1; ++j) {
        for (int i = 1; i < M - 1; ++i) {
            u[j][i] = mean;
            f[j][i] = 0.0f; // No external heat sources for the Laplace problem
        }
    }
}

int main(int argc, char* argv[]) {
    const int N = 512;      // Grid size in the x-direction
    const int M = 512;      // Grid size in the y-direction
    const int iter_max = 10000; // Maximum iterations
    const double tol = 1.0e-4;    // Convergence tolerance
    const double dx = 1.0 / N;    // Grid spacing in x-direction
    const double dy = 1.0 / M;    // Grid spacing in y-direction

    // Boundary conditions
    const double topBC = 100.0;
    const double bottomBC = 0.0;
    const double edgeBC = 100.0;

    // Allocate memory for the grids
    double u[N][M] = {0.0};    // Current values
    double unew[N][M] = {0.0}; // Updated values
    double f[N][M] = {0.0};    // Source term

    // Initialize the grid and boundary values
    double mean = 0.0;
    initialize<N, M>(dx, dy, mean, topBC, bottomBC, edgeBC, u, f);

    std::cout << "Jacobi relaxation calculation: " << N << " x " << M << " grid\n";

    // Perform the Jacobi iteration
    double error = 1.0; // Initial error
    jacobi<N, M>(iter_max, tol, dx, dy, error, f, u, unew);

    // Write results to a file
    std::ofstream outFile("output.txt");
    if (outFile.is_open()) {
        for (int j = 0; j < N; ++j) {
            for (int i = 0; i < M; ++i) {
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
