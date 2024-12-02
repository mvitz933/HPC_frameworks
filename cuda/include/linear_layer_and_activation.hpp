#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include <string>
#include <iostream>


__global__ void linear_layer_and_activation(float *weight_matrix, float *biases, float *x_inputs, 
	                                        float *z_values, float *activation_values, 
											int nr_output_neurons, int nr_input_neurons);