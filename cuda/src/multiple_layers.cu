#include "multiple_layers.hpp"


__global__ void linear_layer_and_activation(float *weight_matrix, float *biases, 
											float *z_values, float *activation_values,
											int* shape, int shape_length){
	int id = threadIdx.x;

	// Define offset for the current layer
	int layer_offset_z_b = 0;
	int layer_offset_weights = 0;
	int layer_offset_activations = 0;
	
	for (int shape_index = 0; shape_index < shape_length; shape_index++){
		// Other threads don't execute anything to avoid out of bounds access
		if (id < shape[shape_index + 1]){
			int nr_inputs_to_this_layer = shape[shape_index];
			// w*x
			for (int neuron_nr = 0; neuron_nr < nr_inputs_to_this_layer; neuron_nr++){
				z_values[layer_offset_z_b + id] += weight_matrix[layer_offset_weights + (nr_inputs_to_this_layer)* id + neuron_nr] *
					activation_values[layer_offset_activations + neuron_nr];
			}

			// w*x + b
			z_values[layer_offset_z_b + id] += biases[layer_offset_z_b + id];

			// sig(w*x + b)	
			// 		                        + shape[shape_index] => write activation values for next layer,
			//                                                      instead of overwriting the input values
			activation_values[layer_offset_activations + shape[shape_index] + id] = 1.0 / (1.0 + exp(-z_values[layer_offset_z_b + id]));
		}

		// Important to do this outside the Memory Guard 
		layer_offset_weights += shape[shape_index] * shape[shape_index + 1];
		layer_offset_z_b += shape[shape_index + 1];
		layer_offset_activations += shape[shape_index];

		// Call syncthreads so we know that all threads have finished working on the current layerbefore we take care of the next layer
		// Try removing this and guess what will happen.
		__syncthreads(); 
	}
}
