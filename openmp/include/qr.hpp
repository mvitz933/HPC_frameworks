#pragma once

#include <vector>

// Perform QR decomposition on matrix A
void qr_decomposition(const std::vector<std::vector<double>>& A,
                      std::vector<std::vector<double>>& Q,
                      std::vector<std::vector<double>>& R);
