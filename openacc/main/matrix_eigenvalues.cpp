#include "eigenvalues.hpp"
#include <iostream>
#include <vector>


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
    // Example symmetric matrix
    constexpr int N = 3;
    const double A[N][N] = {
        {4.0, 1.0, 2.0},
        {1.0, 2.0, 0.0},
        {2.0, 0.0, 3.0}
    };

    double eigenvalues[N];
    int max_iters = 1000;
    double tol = 1e-6;

    try {
        calculate_all_eigenvalues<N>(A, eigenvalues, max_iters, tol);

        // Print the eigenvalues
        print_vector<N>(eigenvalues, "Solution Vector x");

    } catch (const std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
    }

    return 0;
}