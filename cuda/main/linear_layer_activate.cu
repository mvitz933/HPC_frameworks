#include "linear_layer.hpp"
#include "bce_cost.hpp"
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
    // Define input dimensions and initialize the layer
    Shape input_shape(1, 3); // (1 rows, 3 columns, transposed vector)
    Shape weight_shape(3, 1); // shape of weights, resulting in a 1x1 output

    LinearLayer layer("test_layer", weight_shape);

    // Allocate memory for input and output
    Matrix input(input_shape);
    input.allocateMemory();
    input[0] = 0.1f; input[1] = 0.2f; input[2] = 0.3f;
    input.copyHostToDevice();

    // Allocate memory for target
    Matrix target(Shape(1, 1)); // 1x1 target matrix
    target.allocateMemory();
    target[0] = 0.0f;
    target.copyHostToDevice();

    // Print initial weights and biases
    printMatrix(layer.getWeightsMatrix(), "Initial Weights");
    printMatrix(layer.getBiasVector(), "Initial Biases");

    // Perform forward pass
    Matrix& output = layer.forward(input);
    output.copyDeviceToHost();

    // Print forward pass output
    std::cout << "Forward pass output:" << std::endl;
    for (int i = 0; i < output.shape.x * output.shape.y; ++i) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;

    // Calculate BCE loss
    BCECost bce;
    float loss = bce.cost(output, target);
    std::cout << "Binary Cross Entropy Loss: " << loss << std::endl;

    // Calculate gradient of BCE loss
    Matrix dZ(output.shape);
    dZ.allocateMemory();
    bce.dCost(output, target, dZ);

    // Perform backpropagation
    float learning_rate = 0.01f;
    Matrix& dA = layer.backprop(dZ, learning_rate);
    dA.copyDeviceToHost();

    // Print backpropagation output (dA)
    std::cout << "Backpropagation output (dA):" << std::endl;
    for (int i = 0; i < dA.shape.x * dA.shape.y; ++i) {
        std::cout << dA[i] << " ";
    }
    std::cout << std::endl;

    // Print updated weights and biases
    printMatrix(layer.getWeightsMatrix(), "Updated Weights");
    printMatrix(layer.getBiasVector(), "Updated Biases");

    return 0;
}
