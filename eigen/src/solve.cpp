#include <Eigen/Dense>
#include <stdexcept>
#include <vector>

// Solve Ax = b using LU decomposition with Eigen
void solve_linear_system_with_lu(const Eigen::MatrixXd& A, const Eigen::VectorXd& b, Eigen::VectorXd& x) {
    // Ensure matrix A is square and compatible with vector b
    if (A.rows() != A.cols() || A.rows() != b.size()) {
        throw std::invalid_argument("Matrix A must be square and compatible with vector b.");
    }

    // Perform LU decomposition and solve the system
    try {
        x = A.lu().solve(b);
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to solve the linear system: " + std::string(e.what()));
    }
}