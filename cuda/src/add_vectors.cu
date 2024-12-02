#include <stdio.h>
#include "add_vectors.hpp"

__global__ void add_vectors(const double a[N], const double b[N], double c[N], int n) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < n) {
        c[id] = a[id] + b[id];
        if (id < 10) { // Debug print for first 10 threads
            printf("id=%d, a[%d]=%f, b[%d]=%f, c[%d]=%f\n", 
                   id, id, a[id], id, b[id], id, c[id]);
        }
    }
}

