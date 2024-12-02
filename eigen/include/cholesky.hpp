#pragma once

#include <Eigen/Dense>

// Function to perform Cholesky decomposition
// Parameters:
// - A: Input symmetric positive definite matrix (as std::vector<std::vector<double>>)
// - L: Output lower triangular matrix (as std::vector<std::vector<double>>)
void cholesky_decomposition(const Eigen::MatrixXd& A, Eigen::MatrixXd& L) ;
