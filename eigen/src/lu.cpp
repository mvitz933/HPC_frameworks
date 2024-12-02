#include "lu.hpp"
#include <Eigen/Dense>
#include <stdexcept>
#include <iostream>

void lu_decomposition(const Eigen::MatrixXd& A,
                      Eigen::MatrixXd& L,
                      Eigen::MatrixXd& U,
                      Eigen::MatrixXd& P) {
    Eigen::Index n = A.rows();

    // Ensure the matrix is square
    if (n == 0 || A.cols() != n) {
        throw std::invalid_argument("Matrix must be square and non-empty.");
    }

    // Perform LU decomposition using Eigen
    Eigen::FullPivLU<Eigen::MatrixXd> lu(A);

    // Check if the matrix is invertible
    if (!lu.isInvertible()) {
        throw std::runtime_error("Matrix is singular and cannot be decomposed.");
    }

    // Extract the L and U matrices
    Eigen::MatrixXd LU = lu.matrixLU();
    L = Eigen::MatrixXd::Identity(n, n);
    U = LU;

    // Fill the lower triangular part of L and zero out the lower part of U
    for (std::size_t i = 1; i < n; ++i) {
        for (std::size_t j = 0; j < i; ++j) {
            L(i, j) = U(i, j);
            U(i, j) = 0.0; // Clear the lower part of U
        }
    }

    // Extract permutation matrix
    P = lu.permutationP();
}
