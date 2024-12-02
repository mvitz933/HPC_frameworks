#include <iostream>
#include "invert.hpp"
#include <Eigen/Dense>

int main() {
    // Define a square matrix
    Eigen::MatrixXd A(2, 2);
    A << 4, 7,
         2, 6;

    Eigen::MatrixXd inverse;

    try {
        // Compute the inverse
        invert_matrix(A, inverse);

        // Print the original matrix
        std::cout << "Matrix A:\n" << A << std::endl;

        // Print the inverse matrix
        std::cout << "Inverse of A:\n" << inverse << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}
