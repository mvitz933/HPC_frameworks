#include "multiply.hpp"
#include <stdexcept>
#include <omp.h>

std::vector<std::vector<double>> multiply_matrices(const std::vector<std::vector<double>>& A,
                                                   const std::vector<std::vector<double>>& B) {
    // Get dimensions of matrices
    std::size_t rowsA = A.size();
    std::size_t colsA = A[0].size();
    std::size_t rowsB = B.size();
    std::size_t colsB = B[0].size();

    // Check if dimensions are compatible for multiplication
    if (colsA != rowsB) {
        throw std::invalid_argument("Matrix dimensions are not compatible for multiplication.");
    }

    // Initialize the result matrix with zeros
    std::vector<std::vector<double>> C(rowsA, std::vector<double>(colsB, 0.0));

    // Perform matrix multiplication using OpenMP
    #pragma omp parallel for collapse(2)
    for (std::size_t i = 0; i < rowsA; ++i) {
        for (std::size_t j = 0; j < colsB; ++j) {
            for (std::size_t k = 0; k < colsA; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return C;
}
