#include <iostream>
#include <vector>
#include "matrix.hpp"
#include "bce_cost.hpp"
#include "nn_exception.hpp"

// Helper function to initialize a Matrix with data
void initializeMatrix(Matrix& matrix, const std::vector<float>& data) {
    for (size_t i = 0; i < data.size(); ++i) {
        matrix[i] = data[i];
    }
    matrix.copyHostToDevice();
}

int main() {
    // Define the size of the data
    const int size = 10;

    // Create predictions and target data
    std::vector<float> predictions_data = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95};
    std::vector<float> target_data = {0, 0, 1, 0, 1, 0, 1, 1, 1, 0};

    // Create Matrix objects for predictions and targets
    Matrix predictions(size, 1);
    Matrix target(size, 1);
    predictions.allocateMemory();
    target.allocateMemory();

    // Initialize matrices with data
    initializeMatrix(predictions, predictions_data);
    initializeMatrix(target, target_data);

    // Compute the binary cross-entropy cost
    BCECost bce_cost;
    float cost_value = bce_cost.cost(predictions, target);
    std::cout << "Binary Cross-Entropy Cost: " << cost_value << std::endl;

    // Compute the gradient of the binary cross-entropy cost
    Matrix dY(size, 1);
    dY.allocateMemory();
    Matrix dCost_matrix = bce_cost.dCost(predictions, target, dY);
    dCost_matrix.copyDeviceToHost();

    // Print the gradient values
    std::cout << "Gradient of Binary Cross-Entropy Cost: ";
    for (int i = 0; i < size; ++i) {
        std::cout << dCost_matrix[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
