#pragma once

#include <cuda_runtime.h>
// Use preprocessor directive to define N
#ifndef ROWS_A
#define ROWS_A 256 // Default value, if not defined
#endif

#ifndef COLS_A
#define COLS_A 256 // Default value, if not defined
#endif

#ifndef COLS_B
#define COLS_B 256 // Default value, if not defined
#endif

// Ensure N is a valid value
//static_assert(N > 0, "Error: N must be greater than 0");
//static_assert(N % 256 == 0, "Error: N must be divisible by 256 for optimal thread/block configuration");

// Matrices declared with __managed__
__managed__ double A[ROWS_A][COLS_A];
__managed__ double B[COLS_A][COLS_B];
__managed__ double C[ROWS_A][COLS_B];

__global__ void multiply_matrices(const double A[ROWS_A][COLS_A], 
                                  const double B[COLS_A][COLS_B], 
                                  double C[ROWS_A][COLS_B]);
