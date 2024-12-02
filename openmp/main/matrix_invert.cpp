#include <iostream>
#include "invert.hpp"

int main() {
    // Define a square matrix A
    std::vector<std::vector<double>> A = {
        {4, 7},
        {2, 6}
    };

    try {
        // Invert the matrix
        std::vector<std::vector<double>> A_inv = invert_matrix(A);

        // Print the inverse matrix
        std::cout << "Inverse Matrix:" << std::endl;
        for (const auto& row : A_inv) {
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

