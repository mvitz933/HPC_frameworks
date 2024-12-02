#include "cholesky.hpp"
#include <iostream>
#include <vector>


int main() {
    // Example input matrix (symmetric positive definite)
    Eigen::MatrixXd A(3, 3);
    A << 4, 12, -16,
         12, 37, -43,
         -16, -43, 98;

    // Matrix to hold the result
    Eigen::MatrixXd L;

    try {
        // Perform Cholesky decomposition
        cholesky_decomposition(A, L);

        // Print the lower triangular matrix L
        std::cout << "Lower triangular matrix L:" << std::endl;
        std::cout << L << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}