#include <iostream>
#include "matrix.hpp"
#include "nn_exception.hpp"

int main() {
    // Create a Matrix object with dimensions 10x10
    Matrix matrix(10, 10);

    // Allocate memory on both host and device
    matrix.allocateMemory();
    std::cout << "Memory allocated on host and device." << std::endl;

    // Initialize host data
    for (size_t i = 0; i < 100; ++i) {
        matrix[i] = static_cast<float>(i);
    }
    std::cout << "Host data initialized." << std::endl;

    // Copy data from host to device
    matrix.copyHostToDevice();
    std::cout << "Data copied from host to device." << std::endl;

    // Clear host data
    for (size_t i = 0; i < 100; ++i) {
        matrix[i] = 0.0f;
    }
    std::cout << "Host data cleared." << std::endl;

    // Copy data back from device to host
    matrix.copyDeviceToHost();
    std::cout << "Data copied from device to host." << std::endl;

    // Verify the data
    bool success = true;
    for (size_t i = 0; i < 100; ++i) {
        if (matrix[i] != static_cast<float>(i)) {
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "Test passed: Data verification successful." << std::endl;
    } else {
        std::cout << "Test failed: Data verification unsuccessful." << std::endl;
    }

    return 0;
}
