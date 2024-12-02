#pragma once

#include <vector>
// Function to multiply two matrices A and B
template <int ROWS_A, int COLS_A, int COLS_B>
void multiply_matrices(const double A[ROWS_A][COLS_A], const double B[COLS_A][COLS_B], double C[ROWS_A][COLS_B]) {
    // Initialize the result matrix with zeros
    #pragma acc parallel loop collapse(2)
    for (int i = 0; i < ROWS_A; ++i) {
        for (int j = 0; j < COLS_B; ++j) {
            C[i][j] = 0.0;
        }
    }

    // Perform matrix multiplication
    #pragma acc parallel loop collapse(2)
    for (int i = 0; i < ROWS_A; ++i) {
        for (int j = 0; j < COLS_B; ++j) {
            double sum=0.0;
            #pragma acc loop reduction(+:sum) 
            for (int k = 0; k < COLS_A; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

