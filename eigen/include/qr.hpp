#pragma once
#include <Eigen/Dense>


// Perform QR decomposition on matrix A
void qr_decomposition(const Eigen::MatrixXd& A, Eigen::MatrixXd& Q, Eigen::MatrixXd& R);
