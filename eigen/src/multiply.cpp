#include <Eigen/Dense>
#include <vector>
#include <stdexcept>

// Matrix multiplication function using Eigen
void multiply_matrices(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, Eigen::MatrixXd& C) {
    // Check if dimensions are compatible for multiplication
    if (A.cols() != B.rows()) {
        throw std::invalid_argument("Matrix dimensions are not compatible for multiplication.");
    }

    // Resize the output matrix to the correct dimensions
    C.resize(A.rows(), B.cols());

    // Perform matrix multiplication using Eigen
    C = A * B;
}
