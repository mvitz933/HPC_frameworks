#include <iostream>
#include "qr.hpp"
#include <Eigen/Dense>

int main() {
    // Define a matrix A
    Eigen::MatrixXd A(3, 3);
    A << 12, -51, 4,
        6, 167, -68,
        -4, 24, -41;
    Eigen::MatrixXd Q, R;

    try {
        // Perform QR decomposition
        qr_decomposition(A, Q, R);

        // Print the matrices
        std::cout << "Matrix Q:\n" << Q << std::endl;
        std::cout << "Matrix R:\n" << R << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}