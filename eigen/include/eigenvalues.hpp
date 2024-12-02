#pragma once

#include <Eigen/Dense>

// Compute eigenvalues of a matrix using Eigen
Eigen::VectorXd calculate_all_eigenvalues(const Eigen::MatrixXd& A);
