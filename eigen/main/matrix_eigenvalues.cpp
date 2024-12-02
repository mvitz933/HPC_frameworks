#include "eigenvalues.hpp"
#include <iostream>
#include <vector>
#include <Eigen/Dense>

int main() {
    Eigen::MatrixXd A(3, 3);
    A << 4.0, 1.0, 2.0,
         1.0, 2.0, 0.0,
         2.0, 0.0, 3.0;


    try {
        Eigen::VectorXd eigenvalues = calculate_all_eigenvalues(A);
        std::cout << "Eigenvalues:\n" << eigenvalues << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}