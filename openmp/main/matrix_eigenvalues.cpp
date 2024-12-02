#include "eigenvalues.hpp"
#include <iostream>
#include <vector>

int main() {
    // Example symmetric matrix
    std::vector<std::vector<double>> A = {
        {4.0, 1.0, 2.0},
        {1.0, 2.0, 0.0},
        {2.0, 0.0, 3.0}
    };


    try {
        // Calculate eigenvalues
        std::vector<double> eigenvalues = calculate_all_eigenvalues(A);

        // Print eigenvalues
        std::cout << "Eigenvalues:" << std::endl;
        for (const auto& value : eigenvalues) {
            std::cout << value << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}
