#include <iostream>
#include "multiply.hpp"
#include <Eigen/Dense>

int main() {
    // Define two Eigen matrices
    Eigen::MatrixXd A(3, 3);
    Eigen::MatrixXd B(3, 3);

    A << 1, 2, 3,
        4, 5, 6,
        7, 8, 9;

    B << 9, 8, 7,
        6, 5, 4,
        3, 2, 1;

    Eigen::MatrixXd C; // Output matrix

    try {
        // Perform matrix multiplication
        multiply_matrices(A, B, C);

        // Print the result
        std::cout << "Result of A * B:\n" << C << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}