#pragma once
#include <stdio.h>
#include <cuda.h>

#define BLOCK_SIZE_X 192
#define BLOCK_SIZE_Y 1

__global__ void naive_jacobi(float *input, float *output, int N);