#pragma once

#include <vector>

void jacobi(int n, int m, int iter_max, float tol, float dx, float dy, double& error,
                const std::vector<std::vector<float>>& f, 
                std::vector<std::vector<float>>& u, 
                std::vector<std::vector<float>>& unew);


void jacobi_tiled(int n, int m, int iter_max, float tol, float dx, float dy, double& error,
                    const std::vector<std::vector<float>>& f, 
                    std::vector<std::vector<float>>& u, 
                    std::vector<std::vector<float>>& unew, int n_threads); 