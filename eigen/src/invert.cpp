#include "invert.hpp"
#include <Eigen/Dense>
#include <stdexcept>

// Function to compute the inverse of a matrix in place
void invert_matrix(const Eigen::MatrixXd& A, Eigen::MatrixXd& inverse) {
    // Ensure the matrix is square
    if (A.rows() == 0 || A.rows() != A.cols()) {
        throw std::invalid_argument("Matrix must be square and non-empty.");
    }

    // Compute the inverse using Eigen
    Eigen::FullPivLU<Eigen::MatrixXd> lu(A);

    if (!lu.isInvertible()) {
        throw std::runtime_error("Matrix is singular and cannot be inverted.");
    }

    inverse = lu.inverse();
}
