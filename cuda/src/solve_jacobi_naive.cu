#include "jacobi_solver_naive.hpp"


__global__ void naive_jacobi(float *input, float *output, int N){
	register int row = blockIdx.y * blockDim.y + threadIdx.y;
	register int column = blockIdx.x * blockDim.x + threadIdx.x;
	if (row == 0 || row == N-1 || column == 0 || column == N-1 || row >= N || column >= N)
		return;
	register int index = N*row + column;
	output[index] = (input[index - 1] + input[index + 1] + input[index - N] + input[index + N])/4.0;
}