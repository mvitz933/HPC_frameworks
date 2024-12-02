#include <iostream>
#include <cmath>
#include <stdexcept>
#include <iomanip>
#include <cstdlib>
#include <cstring>
#include <openacc.h>
#include "cholesky.hpp"

// Helper Function to Print a Matrix
template <int Rows, int Cols>
void print_matrix(const double matrix[Rows][Cols], const std::string& name) {
    std::cout << name << ":\n";
    for (int i = 0; i < Rows; ++i) {
        for (int j = 0; j < Cols; ++j) {
            std::cout << std::setw(10) << matrix[i][j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}



void print_matrix(const double* matrix, int n, const std::string& name) {
    std::cout << name << ":\n";
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << std::setw(10) << matrix[i * n + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

int main() {
    const int n = 3; // Matrix size
    std::vector<std::vector<double>> A_2d = {
        {4.0, 12.0, -16.0},
        {12.0, 37.0, -43.0},
        {-16.0, -43.0, 98.0}
    };

    double* A_v2 = new double[n * n];
    double* L_v2 = new double[n * n];
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A_v2[i * n + j] = A_2d[i][j];
        }
    }

    // Initialize the output matrix
    memset(L_v2, 0, n * n * sizeof(double));

    CholeskyIntermediate intermediates_v2;

    try {
        // Perform Cholesky decomposition
        cholesky_decomposition_v2(A_v2, L_v2, n, intermediates_v2);
    } catch (const std::runtime_error& e) {
        std::cerr << "Error in cholesky_decomposition_v2: " << e.what() << "\n";
    }

    delete[] A_v2;

    constexpr int N = 3; // Matrix size
    const double A_v1[N][N] = {
        {4.0, 12.0, -16.0},
        {12.0, 37.0, -43.0},
        {-16.0, -43.0, 98.0}
    };

    double L_v1[N][N]; // Initialize L_v1

    memset(L_v1, 0, N * N * sizeof(double));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            L_v1[i][j]=0;
        }
    }
    CholeskyIntermediate intermediates_v1;


    try {
        // Perform Cholesky decomposition
        cholesky_decomposition_v1<N>(A_v1, L_v1, intermediates_v1);
    } catch (const std::runtime_error& e) {
        std::cerr << "Error in cholesky_decomposition_v1: " << e.what() << "\n";
    }

    // Compare sum1 values
    compare_vectors(intermediates_v1.sum1_values, intermediates_v2.sum1_values, "sum1");

    // Compare intermediate L matrices
    compare_matrices(intermediates_v1.L_values_v1, intermediates_v2.L_values_v2, N, "L");
    compare_matrices(intermediates_v1.A_values_v1, intermediates_v2.A_values_v2, N, "A");

    // Optionally, compare final L matrices
    // Print the result
    std::cout << "Lower triangular matrix L from v1:\n";
    print_matrix<N, N>(L_v1, "L_v1");

    std::cout << "Lower triangular matrix L from v2:\n";
    print_matrix(L_v2, n, "L_v2");

    // Clean up
    delete[] L_v2;

    return 0;
}
