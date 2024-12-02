#include <iostream>
#include "qr.hpp"
#include <iostream>
#include <vector>

// Helper Function to Print a Matrix
template <int Rows, int Cols>
void print_matrix(const double (&matrix)[Rows][Cols], const std::string& name) {
    std::cout << name << ":\n";
    for (int i = 0; i < Rows; ++i) {
        for (int j = 0; j < Cols; ++j) {
            std::cout << matrix[i][j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

// Main Function
int main() {
    constexpr int M = 3, N = 3;

    const double A[M][N] = {
        {12, -51, 4},
        {6, 167, -68},
        {-4, 24, -41}
    };

    double Q[M][N] = {0.0}; // Pre-initialize Q
    double R[N][N] = {0.0}; // Pre-initialize R

    try {
        // Perform QR Decomposition
        qr_decomposition<M, N>(A, Q, R);

        // Print Results
        print_matrix(Q, "Matrix Q");
        print_matrix(R, "Matrix R");
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }

    return 0;
}

