#include "solve.hpp"
#include <iostream>
#include <vector>
#include <Eigen/Dense>

int main() {
    // Define matrix A and vector b
    Eigen::MatrixXd A(3, 3);
    Eigen::VectorXd b(3);

    A << 2, -1, 1,
         3, 3, 9,
         3, 3, 5;

    b << 2, -1, 4;

    Eigen::VectorXd x; // Solution vector

    try {
        // Solve the system
        solve_linear_system_with_lu(A, b, x);

        // Print the solution
        std::cout << "Solution vector x:\n" << x << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}