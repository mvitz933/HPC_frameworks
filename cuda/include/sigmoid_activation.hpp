#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cmath>

__global__ void sigmoidActivation(float *z_matrix, float *activation_matrix);