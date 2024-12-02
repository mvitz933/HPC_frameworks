#pragma once
#include <Eigen/Dense>

// Perform LU decomposition on a square matrix A using Eigen
void lu_decomposition(const Eigen::MatrixXd& A,
                      Eigen::MatrixXd& L,
                      Eigen::MatrixXd& U,
                      Eigen::MatrixXd& P);
