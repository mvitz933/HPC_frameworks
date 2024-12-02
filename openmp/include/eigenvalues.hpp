#pragma once
#include <vector>

// Function to compute all eigenvalues of a square matrix
// Input: A (square matrix), max_iters (maximum number of iterations), tol (tolerance for convergence)
// Output: A vector of eigenvalues
std::vector<double> calculate_all_eigenvalues(const std::vector<std::vector<double>>& A, int max_iters = 1000, double tol = 1e-6);
