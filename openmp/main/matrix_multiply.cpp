#include <iostream>
#include "multiply.hpp"

int main() {
    // Define two matrices
    std::vector<std::vector<double>> A = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    std::vector<std::vector<double>> B = {
        {9, 8, 7},
        {6, 5, 4},
        {3, 2, 1}
    };

    try {
        // Multiply matrices
        std::vector<std::vector<double>> C = multiply_matrices(A, B);

        // Print result
        std::cout << "Resulting Matrix:" << std::endl;
        for (const auto& row : C) {
            for (const auto& elem : row) {
                std::cout << elem << " ";
            }
            std::cout << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}
