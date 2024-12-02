#include <stdio.h>
#include <cuda.h>

#define BLOCK_SIZE_X 128
#define BLOCK_SIZE_Y 2

#define iBLOCK_SIZE_X 128
#define iBLOCK_SIZE_Y 4

#define TILE_SIZE_X 4
#define TILE_SIZE_Y 2


__global__ void shmem_jacobi(float *in, float *out, int n);

__global__ void improved_shmem_jacobi(float *in, float *out, int n);