#include <iostream>
#include "qr.hpp"
#include <iostream>
#include <vector>

int main() {
    // Define a matrix A
    std::vector<std::vector<double>> A = {
        {12, -51, 4},
        {6, 167, -68},
        {-4, 24, -41}
    };

    try {
        // Perform QR decomposition
        std::vector<std::vector<double>> Q, R;
        qr_decomposition(A, Q, R);

        // Print Q matrix
        std::cout << "Matrix Q:" << std::endl;
        for (const auto& row : Q) {
            for (const auto& elem : row) {
                std::cout << elem << " ";
            }
            std::cout << std::endl;
        }

        // Print R matrix
        std::cout << "Matrix R:" << std::endl;
        for (const auto& row : R) {
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
