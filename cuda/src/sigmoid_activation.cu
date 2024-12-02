
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cmath>
#include "sigmoid_activation.hpp"

__global__ void sigmoidActivation(float *z_matrix, float *activation_matrix){
    int index = threadIdx.x;
    activation_matrix[index] = 1.0 / (1.0 + exp(-z_matrix[index]));
}

