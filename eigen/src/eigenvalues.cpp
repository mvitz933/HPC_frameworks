#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <stdexcept>

// Function to calculate all eigenvalues of a matrix
Eigen::VectorXd calculate_all_eigenvalues(const Eigen::MatrixXd& A) {
    // Ensure the matrix is square
    if (A.rows() == 0 || A.rows() != A.cols()) {
        throw std::invalid_argument("Matrix must be square and non-empty.");
    }

    // Perform eigenvalue decomposition using Eigen
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(A);

    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Eigen decomposition failed.");
    }

    // Return the eigenvalues as an Eigen vector
    return solver.eigenvalues();
}
