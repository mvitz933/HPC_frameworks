#pragma once
#include <vector>

// Function to perform Cholesky decomposition
// Input: A (square symmetric positive-definite matrix)
// Output: L (lower triangular matrix such that A = L * L^T)
void cholesky_decomposition(const std::vector<std::vector<double>>& A,
                            std::vector<std::vector<double>>& L);
