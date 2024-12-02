#include <iostream>
#include "lu.hpp"

int main() {
    std::vector<std::vector<double>> A = {
        {4, 3},
        {6, 3}
    };

    try {
        std::vector<std::vector<double>> L, U, P;
        lu_decomposition(A, L, U, P);

        std::cout << "Permutation Matrix P:" << std::endl;
        for (const auto& row : P) {
            for (const auto& elem : row) {
                std::cout << elem << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "Lower Triangular Matrix L:" << std::endl;
        for (const auto& row : L) {
            for (const auto& elem : row) {
                std::cout << elem << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "Upper Triangular Matrix U:" << std::endl;
        for (const auto& row : U) {
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
