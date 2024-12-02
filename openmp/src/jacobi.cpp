#include <omp.h>
#include "jacobi.hpp"
#include "timer.hpp"
#include <vector>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>


void jacobi(int n, int m, int iter_max, float tol, float dx, float dy, double& error,
                const std::vector<std::vector<float>>& f, 
                std::vector<std::vector<float>>& u, 
                std::vector<std::vector<float>>& unew) {
    //float error = 1.0f;
    int iter = 0;

    // Start the timer
    StartTimer();

    while (error > tol && iter < iter_max) {
        error = 0.0;

        #pragma omp parallel
        {
            // Update loop for new values with source term f
            #pragma omp for reduction(max:error) collapse(2)
            for (int j = 1; j < n - 1; j++) {
                for (int i = 1; i < m - 1; i++) {
                    unew[j][i] = 0.25f * (u[j][i + 1] + u[j][i - 1]
                                       + u[j - 1][i] + u[j + 1][i]
                                       - f[j][i] * dx * dy);

                    // Track the maximum error
                    error = fmaxf(error, fabsf(unew[j][i] - u[j][i]));
                }
            }

            // Update u with new values from unew
            #pragma omp for collapse(2)
            for (int j = 1; j < n - 1; j++) {
                for (int i = 1; i < m - 1; i++) {
                    u[j][i] = unew[j][i];
                }
            }
        }

        if (iter % 100 == 0) {
            std::cout << iter << ", " << error << "\n";
        }

        iter++;
    }

    // Stop the timer and display runtime
    double runtime = GetTimer();
    std::cout << "Total time: " << runtime / 1000.0f << " s\n";
}



/*
If I want to use RMS instead of max error I will need to do this logic.
Need to decide whether to pass as a param which method to use.
float error_sum = 0.0f;
#pragma omp parallel
{
    #pragma omp for reduction(+:error_sum) collapse(2)
    for (int j = 1; j < n - 1; j++) {
        for (int i = 1; i < m - 1; i++) {
            unew[j][i] = 0.25f * (u[j][i + 1] + u[j][i - 1]
                               + u[j - 1][i] + u[j + 1][i]
                               - f[j][i] * dx * dy);

            // Sum squared errors
            float local_error = unew[j][i] - u[j][i];
            error_sum += local_error * local_error;
        }
    }
}

// Calculate RMS error
error = sqrt(error_sum / ((n - 2) * (m - 2)));

*/
void jacobi_tiled(int n, int m, int iter_max, float tol, float dx, float dy, double& error,
                    const std::vector<std::vector<float>>& f, 
                    std::vector<std::vector<float>>& u, 
                    std::vector<std::vector<float>>& unew, int n_threads){
    int iter = 0;

    // Determine the number of threads and calculate tile dimensions
    int tile_width, tile_height;

    tile_width = n / (n_threads / 2);
    tile_height = m / (n_threads / 2);

    while (error > tol && iter < iter_max) {
        error = 0.0;

        #pragma omp parallel shared(u, unew, f, tile_width, tile_height, error)
        {
            #pragma omp single nowait
            {
                // Iterate over tiles
                for (int i = 1; i < n; i += tile_width) {
                    for (int j = 1; j < m; j += tile_height) {
                        #pragma omp task shared(u, unew, f, error) firstprivate(i, j)
                        {
                            // Process each tile
                            for (int k = i; k < std::min(i + tile_width, n - 1); ++k) {
                                for (int l = j; l < std::min(j + tile_height, m - 1); ++l) {
                                    unew[k][l] = 0.25f * (u[k][l + 1] + u[k][l - 1] 
                                                        + u[k - 1][l] + u[k + 1][l] 
                                                        - f[k][l] * dx * dy);


                                    #pragma omp critical
                                    {
                                        double diff = fabs(unew[j][i] - u[j][i]);
                                        error = std::max(error, diff);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            #pragma omp taskwait
        }

        // Copy unew back into u
        #pragma omp parallel for collapse(2)
        for (int i = 1; i < n - 1; ++i) {
            for (int j = 1; j < m - 1; ++j) {
                u[i][j] = unew[i][j];
            }
        }

        if (iter % 100 == 0) {
            std::cout << "Iteration " << iter << ", Error: " << error << std::endl;
        }

        iter++;
    }

    std::cout << "Jacobi Tiled completed in " << iter << " iterations." << std::endl;
}