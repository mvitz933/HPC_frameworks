#include <iostream>
#include "solve.hpp"

int main() {
    // Define matrix A and vector b for the system Ax = b
    std::vector<std::vector<double>> A = {
        {2, -1, 1},
        {3, 3, 9},
        {3, 3, 5}
    };
    std::vector<double> b = {2, -1, 4};

    try {
        // Solve the system using LU decomposition
        std::vector<double> x = solve_linear_system_with_lu(A, b);

        // Print the solution
        std::cout << "Solution:" << std::endl;
        for (const auto& xi : x) {
            std::cout << xi << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}