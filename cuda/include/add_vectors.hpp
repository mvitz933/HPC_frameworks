#pragma once

#include <cuda_runtime.h>
// Use preprocessor directive to define N
#ifndef N
#define N 2560 // Default value, if not defined
#endif

// Ensure N is a valid value
static_assert(N > 0, "Error: N must be greater than 0");
static_assert(N % 256 == 0, "Error: N must be divisible by 256 for optimal thread/block configuration");

// Global pointers for managed memory (declaration)
__managed__ double vector_a[N];
__managed__ double vector_b[N];
__managed__ double vector_c[N];

// Kernel function declaration
__global__ void add_vectors(const double *__restrict__ a, const double *__restrict__ b, double *__restrict__ c, int n);

