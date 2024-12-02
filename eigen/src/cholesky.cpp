#include <Eigen/Dense>
#include <vector>
#include <stdexcept>
#include <iostream>

// Function to perform Cholesky decomposition using Eigen
void cholesky_decomposition(const Eigen::MatrixXd& A, Eigen::MatrixXd& L) {
    // Ensure the input matrix is square
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("Matrix must be square.");
    }

    // Perform Cholesky decomposition using Eigen
    Eigen::LLT<Eigen::MatrixXd> llt(A);

    // Check for positive definiteness
    if (llt.info() == Eigen::NumericalIssue) {
        throw std::runtime_error("Matrix is not positive definite.");
    }

    // Assign the lower triangular matrix to L
    L = llt.matrixL();
}
