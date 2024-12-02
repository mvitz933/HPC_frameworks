#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include <string>
#include <iostream>
#include <algorithm>


#pragma once
#ifdef __INTELLISENSE__
void __syncthreads();
#endif


__global__ void multiple_inputs(float *weight_matrix, float *biases, float *z_values, 
								float *activation_values, int* shape, int shape_length);