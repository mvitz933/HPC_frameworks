#pragma once
#include <vector>

struct CholeskyIntermediate {
    std::vector<double> sum1_values;
    std::vector<std::vector<std::vector<double>>> L_values_v1; // For cholesky_decomposition_v1
    std::vector<std::vector<std::vector<double>>> A_values_v1; // For cholesky_decomposition_v1
    std::vector<std::vector<double>> L_values_v2;              // For cholesky_decomposition_v2
    std::vector<std::vector<double>> A_values_v2;              // For cholesky_decomposition_v2
    // You can also add sum2_values or other variables as needed
};


void compare_matrices(const std::vector<std::vector<std::vector<double>>>& L_v1,
                        const std::vector<std::vector<double>>& L_v2_flat,
                        int N, const std::string& var_name) {
    size_t num_iterations = std::min(L_v1.size(), L_v2_flat.size());

    for (size_t iter = 0; iter < num_iterations; ++iter) {
        std::cout << "Comparison of " + var_name + " matrices after iteration " << iter << ":\n";

        const auto& L1 = L_v1[iter];
        const auto& L2_flat = L_v2_flat[iter];

        // Convert flattened L2 to 2D vector
        std::vector<std::vector<double>> L2(N, std::vector<double>(N));
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                L2[i][j] = L2_flat[i * N + j];
            }
        }

        // Compare L1 and L2
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                double diff = L1[i][j] - L2[i][j];
                std::cout << var_name+"[" << i << "][" << j << "]: v1 = " << L1[i][j]
                          << ", v2 = " << L2[i][j]
                          << ", Difference = " << diff << "\n";
            }
        }
        std::cout << "\n";
    }
}


void compare_vectors(const std::vector<double>& vec1, const std::vector<double>& vec2, const std::string& var_name) {
    std::cout << "Comparing " << var_name << " values:\n";
    size_t n = std::min(vec1.size(), vec2.size());
    for (size_t i = 0; i < n; ++i) {
        double diff = vec1[i] - vec2[i];
        std::cout << "Step " << i << ": " << var_name << "_v1 = " << vec1[i]
                  << ", " << var_name << "_v2 = " << vec2[i]
                  << ", Difference = " << diff << "\n";
    }
    std::cout << "\n";
}



// Inline function for read-only access to 2D array
inline const double& A_2D(const double* array, int i, int j, int cols) {
    return array[i * cols + j];
}

// Inline function for writable 2D array access (only for L)
inline double& A_2D(double* array, int i, int j, int cols) {
    return array[i * cols + j];
}


// Function to perform Cholesky decomposition
// Input: A (square symmetric positive-definite matrix)
// Output: L (lower triangular matrix such that A = L * L^T)
template <int N>
void cholesky_decomposition_v1(const double A[N][N], double L[N][N], CholeskyIntermediate& intermediates) {
    // Ensure the input matrix is square
    static_assert(N > 0, "Matrix size must be greater than 0.");

    // Error flags
    bool is_error = false;
    int error_index = -1;

    // Cholesky decomposition logic
    for (int i = 0; i < N; ++i) {
        double sum1 = 0.0;
        // Store a copy of A after each iteration
        std::vector<std::vector<double>> A_copy(N, std::vector<double>(N));
        for (int a = 0; a < N; ++a) {
            for (int b = 0; b < N; ++b) {
                A_copy[a][b] = A[a][b];
            }
        }
        intermediates.A_values_v1.push_back(A_copy);
        // Compute diagonal element L[i][i]
        //#pragma acc parallel loop reduction(+:sum1)
        for (int k = 0; k < i; ++k) {
            sum1 += L[i][k] * L[i][k];
        }
        intermediates.sum1_values.push_back(sum1); // Store sum1

        double diag = A[i][i] - sum1;

        // Error handling on device
        if (diag < 0.0) {
            is_error = true;
            error_index = i;
            break;
        }

        L[i][i] = sqrt(diag);

        // Compute off-diagonal elements L[j][i] for j > i
        for (int j = i + 1; j < N; ++j) {
            double sum2 = 0.0;

            //#pragma acc parallel loop reduction(+:sum2)
            for (int k = 0; k < i; ++k) {
                sum2 += L[j][k] * L[i][k];
            }

            L[j][i] = (A[j][i] - sum2) / L[i][i];
        }

        // Store a copy of L after each iteration
        std::vector<std::vector<double>> L_copy(N, std::vector<double>(N));
        for (int a = 0; a < N; ++a) {
            for (int b = 0; b < N; ++b) {
                L_copy[a][b] = L[a][b];
            }
        }
        intermediates.L_values_v1.push_back(L_copy);

    }


    // Check if an error occurred
    if (is_error) {
        throw std::runtime_error("Matrix is not positive definite at row: " + std::to_string(error_index));
    }
}



void cholesky_decomposition_v2(const double* A, double* L, int n, CholeskyIntermediate& intermediates) {
    // Error flags
    bool is_error = false;
    int error_index = -1;

    // Initialize L to zero (if not initialized elsewhere)
    memset(L, 0, n * n * sizeof(double));

    // Create data region encompassing all operations
        for (int i = 0; i < n; ++i) {
            double sum1 = 0.0;
            std::vector<double> A_copy(A, A + n * n);
            intermediates.A_values_v2.push_back(A_copy);
            // Compute diagonal element L[i][i]
            #pragma acc parallel loop reduction(+:sum1)
            for (int k = 0; k < i; ++k) {
                sum1 += L[i * n + k] * L[i * n + k];
            }
            intermediates.sum1_values.push_back(sum1); // Store sum1

            double diag = A[i * n + i] - sum1;
            if (diag < 0.0) {
                is_error = true;
                error_index = i;
                break;
            }
            L[i * n + i] = sqrt(diag);

            // Compute off-diagonal elements L[j][i] for j > i
            for (int j = i + 1; j < n; ++j) {
                double sum2 = 0.0;

                #pragma acc parallel loop reduction(+:sum2)
                for (int k = 0; k < i; ++k) {
                    sum2 += L[j * n + k] * L[i * n + k];
                }

                L[j * n + i] = (A[j * n + i] - sum2) / L[i * n + i];
            }

            // Store a copy of L after each iteration
            std::vector<double> L_copy(L, L + n * n);
            intermediates.L_values_v2.push_back(L_copy);
        }

    // Check if an error occurred
    if (is_error) {
        throw std::runtime_error("Some error occurred at row: " + std::to_string(error_index));
    }
}
