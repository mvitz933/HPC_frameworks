#include <iostream>
#include "multiply.hpp"
#include <fstream>

// Function to read a matrix from a file
template <int ROWS, int COLS>
bool read_matrix_from_file(const std::string& filename, double matrix[ROWS][COLS]) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: Cannot open file '" << filename << "'\n";
        return false;
    }

    for (int i = 0; i < ROWS; ++i) {
        for (int j = 0; j < COLS; ++j) {
            if (!(infile >> matrix[i][j])) {
                std::cerr << "Error: Not enough data in file for a " << ROWS << "x" << COLS << " matrix\n";
                return false;
            }
        }
    }

    infile.close();
    return true;
}


int main() {
    // Define matrix dimensions
    constexpr int ROWS_A = 3;
    constexpr int COLS_A = 3;
    constexpr int ROWS_B = 3;
    constexpr int COLS_B = 3;

    // Check if dimensions are valid for multiplication
    static_assert(COLS_A == ROWS_B, "Matrix dimensions are not compatible for multiplication.");

    // Declare matrices
    double A[ROWS_A][COLS_A];
    double B[COLS_A][COLS_B];
    double C[ROWS_A][COLS_B] = {0.0}; // Result matrix

    // Read matrices from files
    if (!read_matrix_from_file<ROWS_A, COLS_A>("matrix_A.txt", A)) {
        return 1;
    }
    if (!read_matrix_from_file<COLS_A, COLS_B>("matrix_B.txt", B)) {
        return 1;
    }

    // Perform matrix multiplication
    multiply_matrices<ROWS_A, COLS_A, COLS_B>(A, B, C);

    // Print the result matrix
    std::cout << "Result matrix C:\n";
    for (int i = 0; i < ROWS_A; ++i) {
        for (int j = 0; j < COLS_B; ++j) {
            std::cout << C[i][j] << " ";
        }
        std::cout << "\n";
    }

    return 0;
}