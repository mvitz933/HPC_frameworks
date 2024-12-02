#pragma once
#include <Eigen/Dense>

// Solve Ax = b using LU decomposition with Eigen
void solve_linear_system_with_lu(const Eigen::MatrixXd& A, const Eigen::VectorXd& b, Eigen::VectorXd& x);

