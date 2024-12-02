#pragma once

#include <vector>
#include <stdexcept>

// Solve Ax = b using LU decomposition
std::vector<double> solve_linear_system_with_lu(const std::vector<std::vector<double>>& A,
                                                const std::vector<double>& b);
