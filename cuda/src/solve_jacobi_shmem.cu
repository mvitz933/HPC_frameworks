#include "jacobi_solver_shmem.hpp"

__global__ void shmem_jacobi(float *in, float *out, int n){
	int local_row = threadIdx.y + 1;
	int local_col = threadIdx.x + 1;
	__shared__ float local_matrix[BLOCK_SIZE_Y + 2][BLOCK_SIZE_X + 2];
	int row = blockIdx.y * blockDim.y + threadIdx.y + 1;
	int col = blockIdx.x * blockDim.x + threadIdx.x + 1;
	int index = n*row + col;
	if (row >= n || col >= n)
		return;
	local_matrix[local_row][local_col] = in[index];
	if (row == n-1 || col == n-1)
		return;
	if (local_row == 1)
		local_matrix[local_row - 1][local_col] = in[n * (row - 1) + col];
	if (local_col == 1)
		local_matrix[local_row][local_col - 1] = in[n * row + col - 1];
	if (local_row == BLOCK_SIZE_Y)
		local_matrix[local_row + 1][local_col] = in[n * (row + 1) + col];
	if (local_col == BLOCK_SIZE_X)
		local_matrix[local_row][local_col + 1] = in[n * row + col + 1];
	__syncthreads();
	out[index] = (local_matrix[local_row][local_col-1] + local_matrix[local_row-1][local_col] + local_matrix[local_row][local_col+1] + local_matrix[local_row+1][local_col])/4.0;
}


__global__ void shmem_improved_jacobi(float *in, float *out, int n){
    int index, local_row, local_col;
	__shared__ float local_matrix[TILE_SIZE_Y * iBLOCK_SIZE_Y + 2][TILE_SIZE_X * iBLOCK_SIZE_Y + 2];

	int tile_threads_x = blockDim.x * TILE_SIZE_X;
	int tile_threads_y = blockDim.y * TILE_SIZE_Y;
	int tile_zero_row = blockIdx.y * tile_threads_y;
	int next_tile_first_row = tile_zero_row + 1 + tile_threads_y; 
	int tile_zero_col = blockIdx.x * tile_threads_x;
	int next_tile_first_col = tile_zero_col + 1 + tile_threads_x;

	local_row = threadIdx.y + 1;
	for (int row = tile_zero_row + local_row; ((row < n) && (row < tile_zero_row + 1 + tile_threads_y)); row += blockDim.y) {
		local_col = threadIdx.x + 1;
		for (int col = tile_zero_col + local_col; ((col < n) && (col < tile_zero_col + 1 + tile_threads_x)); col += blockDim.x) {
			index = n*row + col;
			local_matrix[local_row][local_col] = in[index];
			if (row == n - 1 || col == n - 1) {
				local_col += blockDim.x;
				continue;
			}
			if (row == tile_zero_row + 1) {
				local_matrix[local_row - 1][local_col] = in[n * tile_zero_row + col];
			}
			if (col == tile_zero_col + 1) {
				local_matrix[local_row][local_col - 1] = in[n * row + tile_zero_col];
			}
			if (row == next_tile_first_row - 1) {
				local_matrix[local_row + 1][local_col] = in[n * next_tile_first_row + col];
			}
			if (col == next_tile_first_col - 1) {
				local_matrix[local_row][local_col + 1] = in[n * row + next_tile_first_col];
			}
			local_col += blockDim.x;
		}
		local_row += blockDim.y;
	}
	__syncthreads();

	local_row = threadIdx.y + 1;
	for (int row = tile_zero_row + local_row; ((row < n - 1) && (row < tile_zero_row + 1 + tile_threads_y)); row += blockDim.y) {
		local_col = threadIdx.x + 1;
		for (int col = tile_zero_col + local_col; ((col < n - 1) && (col < tile_zero_col + 1 + tile_threads_x)); col += blockDim.x) {
			index = n*row + col;
			out[index] = (local_matrix[local_row][local_col-1] + local_matrix[local_row-1][local_col] + local_matrix[local_row][local_col+1] + local_matrix[local_row+1][local_col])/4.0;
			local_col += blockDim.x;
		}
		local_row += blockDim.y;
	}
}

