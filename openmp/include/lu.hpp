#pragma once

#include <vector>

// Perform LU decomposition on a square matrix A
void lu_decomposition(const std::vector<std::vector<double>>& A,
                      std::vector<std::vector<double>>& L,
                      std::vector<std::vector<double>>& U,
                      std::vector<std::vector<double>>& P);