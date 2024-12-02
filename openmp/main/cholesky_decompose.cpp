#include "cholesky.hpp"
#include <iostream>
#include <vector>

void print_matrix(const std::vector<std::vector<double>>& mat) {
    for (const auto& row : mat) {
        for (double val : row) {
            std::cout << val << " ";
        }
        std::cout << "\n";
    }
}

int main() {
    // Example symmetric positive-definite matrix
    std::vector<std::vector<double>> A = {
        {4.0, 12.0, -16.0},
        {12.0, 37.0, -43.0},
        {-16.0, -43.0, 98.0}
    };

    std::vector<std::vector<double>> L;

    try {
        cholesky_decomposition(A, L);

        std::cout << "Input matrix A:\n";
        print_matrix(A);

        std::cout << "Lower triangular matrix L:\n";
        print_matrix(L);
    } catch (const std::runtime_error& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }

    return 0;
}
