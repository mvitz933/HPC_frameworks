#include <iostream>
#include <Eigen/Dense>
#include "lu.hpp" // Ensure lu.hpp is updated to accept and return Eigen matrices

int main() {
    // Define a square matrix A using Eigen
    Eigen::MatrixXd A(2, 2);
    A << 4, 3,
         6, 3;

    try {
        // Perform LU decomposition
        Eigen::MatrixXd L, U, P;
        lu_decomposition(A, L, U, P);

        // Print the permutation matrix P
        std::cout << "Permutation matrix P:\n" << P << "\n\n";

        // Print the lower triangular matrix L
        std::cout << "Lower triangular matrix L:\n" << L << "\n\n";

        // Print the upper triangular matrix U
        std::cout << "Upper triangular matrix U:\n" << U << "\n\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}
