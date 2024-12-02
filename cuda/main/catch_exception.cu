#include "nn_exception.hpp"
#include <cuda_runtime.h>

int main() {
    // Allocate memory on the GPU
    float* d_data;
    cudaError_t error = cudaMalloc((void**)&d_data, 100 * sizeof(float));

    // Check for CUDA errors and throw an exception if any
    try {
        NNException::throwIfDeviceErrorsOccurred("Failed to allocate GPU memory");
    } catch (const NNException& e) {
        std::cerr << "Caught NNException: " << e.what() << std::endl;
        return -1; // Return an error code
    }

    // Free the GPU memory
    error = cudaFree(d_data);

    // Check for CUDA errors again
    try {
        NNException::throwIfDeviceErrorsOccurred("Failed to free GPU memory");
    } catch (const NNException& e) {
        std::cerr << "Caught NNException: " << e.what() << std::endl;
        return -1; // Return an error code
    }

    std::cout << "CUDA operations completed successfully" << std::endl;
    return 0; // Return success
}
