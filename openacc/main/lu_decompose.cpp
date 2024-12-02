#include "lu.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <iomanip>
#include <cstring>

// Helper Function to Print a Matrix
template <int Rows, int Cols>
void print_matrix(const double matrix[Rows][Cols], const std::string& name) {
    std::cout << name << ":\n";
    for (int i = 0; i < Rows; ++i) {
        for (int j = 0; j < Cols; ++j) {
            std::cout << std::setw(10) << matrix[i][j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}



// Function to read a matrix from a file
template <int N>
bool read_matrix_from_file(const std::string& filename, double matrix[N][N]) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: Cannot open file '" << filename << "'\n";
        return false;
    }

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (!(infile >> matrix[i][j])) {
                std::cerr << "Error: Not enough data in file for a " << N << "x" << N << " matrix\n";
                return false;
            }
        }
    }

    infile.close();
    return true;
}


int main() {
    constexpr int N = 2;
    double A[N][N];
    double L[N][N], U[N][N], P[N][N];

    // Read the matrix from a file
    std::string filename = "lu_matrix.txt";
    if (!read_matrix_from_file<N>(filename, A)) {
        return 1; // Exit if reading fails
    }

    try {
        lu_decomposition<N>(A, L, U, P);

        // Print L, U, and P matrices
        print_matrix<N,N>(L, "Matrix L");
        print_matrix<N,N>(U, "Matrix U");
        print_matrix<N,N>(P, "Permutation Matrix P");
    } catch (const std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
    }

    return 0;
}
