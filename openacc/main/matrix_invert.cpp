#include <iostream>
#include "invert.hpp"
#include <iomanip>
#include <stdexcept>

// Include the invert_matrix function here
// ...

// Function to print a matrix
template <int N>
void print_matrix(const double matrix[N][N], const std::string& name) {
    std::cout << name << ":\n";
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << std::setw(10) << matrix[i][j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

int main() {
    constexpr int N = 2;
    double A[N][N] = {
        {4, 7},
        {2, 6}
    };
    double inverse[N][N];

    try {
        invert_matrix<N>(A, inverse);

        // Print the inverse matrix
        print_matrix<N>(inverse, "Inverse Matrix");
    } catch (const std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
    }

    return 0;
}
