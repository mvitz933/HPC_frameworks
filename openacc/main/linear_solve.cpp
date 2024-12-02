#include <iostream>
#include "solve.hpp"
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include "lu.hpp"
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



// Function to print a vector
template <int N>
void print_vector(const double vec[N], const std::string& name) {
    std::cout << name << ":\n";
    for (int i = 0; i < N; ++i) {
        std::cout << vec[i] << "\n";
    }
    std::cout << "\n";
}

int main() {
    constexpr int N = 3;
    double A[N][N] = {
        {2, -1, 1},
        {3, 3, 9},
        {3, 3, 5}
    };
    double b[N] = {2, -1, 4};
    double x[N];

    try {
        solve_linear_system_with_lu<N>(A, b, x);

        // Print the solution vector x
        print_vector<N>(x, "Solution Vector x");
    } catch (const std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
    }

    return 0;
}
