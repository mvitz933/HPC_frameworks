#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include <string>
#include <iostream>
#include "linear_layer_and_activation.hpp"

__global__ void linear_layer_and_activation(float *weight_matrix, float *biases, float *x_inputs, 
	                                        float *z_values, float *activation_values, 
											int nr_output_neurons, int nr_input_neurons){
	int id = threadIdx.x;

	// w*x
	for (int neuron_nr = 0; neuron_nr < nr_input_neurons; neuron_nr++){
		z_values[id] += weight_matrix[(nr_input_neurons)* id + neuron_nr] * x_inputs[neuron_nr];
	}

	// w*x + b
	z_values[id] += biases[id];

	// sig(w*x + b)
	activation_values[id] = 1.0 / (1.0 + exp(-z_values[id]));
	
}
