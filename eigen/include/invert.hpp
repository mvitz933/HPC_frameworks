#pragma once
#include <Eigen/Dense>

// Invert a matrix using Eigen
void invert_matrix(const Eigen::MatrixXd& A, Eigen::MatrixXd& inverse);

