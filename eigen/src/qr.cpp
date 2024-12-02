#include "qr.hpp"
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <omp.h>


void qr_decomposition(const Eigen::MatrixXd& A, Eigen::MatrixXd& Q, Eigen::MatrixXd& R) {
    // Ensure the input matrix A is non-empty
    if (A.rows() < A.cols()) {
        throw std::invalid_argument("Matrix A must have at least as many rows as columns.");
    }

    // Perform QR decomposition using Eigen's HouseholderQR
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(A);

    // Extract Q and R matrices
    Q = qr.householderQ() * Eigen::MatrixXd::Identity(A.rows(), A.cols());
    R = qr.matrixQR().topRows(A.cols()).triangularView<Eigen::Upper>();
}
