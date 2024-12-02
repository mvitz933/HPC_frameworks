#include "relu_activation.hpp"
#include "nn_exception.hpp"
#include "matrix.hpp"

void printMatrix(Matrix& matrix, const std::string& name) {
    matrix.copyDeviceToHost();
    std::cout << name << ":" << std::endl;
    for (int i = 0; i < matrix.shape.x * matrix.shape.y; ++i) {
        std::cout << matrix[i] << " ";
    }
    std::cout << std::endl;
}

int main() {
    // Define input dimensions and initialize the matrix
    Shape input_shape(1, 3); // (1 rows, 3 columns)

    // Initialize ReLUActivation
    ReLUActivation relu("relu_activation");

    // Allocate memory for input matrix
    Matrix input(input_shape);
    input.allocateMemory();
    input[0] = -1.0f; input[1] = 0.0f; input[2] = 1.0f;
    input.copyHostToDevice();

    // Perform forward pass
    Matrix& output = relu.forward(input);
    output.copyDeviceToHost();

    // Print forward pass output
    printMatrix(output, "Forward pass output");

    // Allocate memory for gradient matrix
    Matrix dA(output.shape);
    dA.allocateMemory();
    dA[0] = 0.1f; dA[1] = 0.2f; dA[2] = 0.3f;
    dA.copyHostToDevice();

    // Perform backward pass
    Matrix& dZ = relu.backprop(dA, 0.01f);
    dZ.copyDeviceToHost();

    // Print backward pass output
    printMatrix(dZ, "Backward pass output");

    return 0;
}
