#include <mpi.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <stdexcept>
#include <iterator>

// Helper function: Print a matrix
void print_matrix(const std::vector<double>& mat, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << mat[i * cols + j] << " ";
        }
        std::cout << "\n";
    }
}

// Compute determinant of a matrix
double compute_determinant(const std::vector<double>& mat, int n) {
    double det = 1.0;
    std::vector<double> temp = mat;

    for (int i = 0; i < n; ++i) {
        if (std::abs(temp[i * n + i]) < 1e-9) return 0.0;  // Singular matrix
        for (int j = i + 1; j < n; ++j) {
            double factor = temp[j * n + i] / temp[i * n + i];
            for (int k = 0; k < n; ++k) {
                temp[j * n + k] -= factor * temp[i * n + k];
            }
        }
        det *= temp[i * n + i];
    }
    return det;
}

// Compute inverse of a block using Gauss-Jordan elimination with pivoting
void invert_block(std::vector<double>& block, int block_size) {
    int n = block_size;
    std::vector<double> identity(n * n, 0.0);

    for (int i = 0; i < n; ++i) {
        identity[i * n + i] = 1.0;
    }

    for (int i = 0; i < n; ++i) {
        int pivot = i;
        for (int j = i + 1; j < n; ++j) {
            if (std::abs(block[j * n + i]) > std::abs(block[pivot * n + i])) {
                pivot = j;
            }
        }
        if (pivot != i) {
            for (int j = 0; j < n; ++j) {
                std::swap(block[i * n + j], block[pivot * n + j]);
                std::swap(identity[i * n + j], identity[pivot * n + j]);
            }
        }

        double diag = block[i * n + i];
        if (std::abs(diag) < 1e-9) {
            throw std::runtime_error("Matrix is singular or nearly singular.");
        }

        for (int j = 0; j < n; ++j) {
            block[i * n + j] /= diag;
            identity[i * n + j] /= diag;
        }

        for (int k = 0; k < n; ++k) {
            if (k != i) {
                double factor = block[k * n + i];
                for (int j = 0; j < n; ++j) {
                    block[k * n + j] -= factor * block[i * n + j];
                    identity[k * n + j] -= factor * identity[i * n + j];
                }
            }
        }
    }

    block = identity;
}


void load_matrix_from_file(const std::string& filename, std::vector<double>& mat, int& rows, int& cols) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + filename);
    }

    std::string line;
    std::vector<double> temp_mat;
    int num_cols = -1;

    while (std::getline(file, line)) {
        // Remove commas and parse the row
        for (char& c : line) {
            if (c == ',') c = ' ';
        }

        std::istringstream iss(line);
        std::vector<double> row_values((std::istream_iterator<double>(iss)), std::istream_iterator<double>());

        if (num_cols == -1) {
            num_cols = row_values.size();
        } else if (row_values.size() != num_cols) {
            throw std::runtime_error("Inconsistent number of columns in matrix file.");
        }

        temp_mat.insert(temp_mat.end(), row_values.begin(), row_values.end());
    }

    file.close();

    if (num_cols == -1) {
        throw std::runtime_error("Empty matrix file.");
    }

    rows = temp_mat.size() / num_cols;
    cols = num_cols;
    mat = std::move(temp_mat);
}
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    int N;  // Rows/columns of the matrix
    int block_size;
    std::vector<double> matrix;
    std::vector<double> local_block;
//    std::cout << "Dimensions are: N \n" << N << "rank:\n " << rank << "size:\n " << size << "block_size:\n " << block_size << "\n";


    if (rank == 0) {
        try {
            // Load matrix from file
            load_matrix_from_file("/home/main/Documents/PROJECTS/HPC/data/matrices/randomMatrix_50.txt", matrix, N, N);            
            std::cout << "Matrix loaded from file. Original dimensions: " << N << "x" << N << "\n";
            std::cout << "Dimensions are: N " << N << "\n rank: " << rank << "\n size: " << size << "\n";

            // Adjust matrix size if necessary
            if (N % size != 0) {
                int padding = size - (N % size);
                int new_N = N + padding;

                std::vector<double> padded_matrix(new_N * new_N, 0.0);
                for (int i = 0; i < N; ++i) {
                    for (int j = 0; j < N; ++j) {
                        padded_matrix[i * new_N + j] = matrix[i * N + j];
                    }
                }

                matrix = std::move(padded_matrix);
                N = new_N;
                std::cout << "Matrix padded to dimensions: " << N << "x" << N << "\n";
            }

            double det = compute_determinant(matrix, N);
            if (std::abs(det) < 1e-9) {
                std::cerr << "Matrix is singular or nearly singular. Cannot compute inverse.\n";
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
        } catch (const std::exception& ex) {
            std::cerr << ex.what() << "\n";
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    std::cout << "About to broadcast\n";

    // Broadcast the adjusted matrix size
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (N % size != 0) {
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);  // Safety check
    }

    block_size = N / size;
    local_block.resize(block_size * N, 0.0);

    // Scatter the matrix to all processes
    if (rank == 0) {
        std::cout << "Scattering data to all processes...\n";
    }

    std::cout << "About to scatter\n";
    MPI_Scatter(matrix.data(), block_size * N, MPI_DOUBLE,
                local_block.data(), block_size * N, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Process local blocks
    for (int i = 0; i < block_size; ++i) {
        std::vector<double> row(local_block.begin() + i * N, local_block.begin() + (i + 1) * N);
        invert_block(row, N);
        std::copy(row.begin(), row.end(), local_block.begin() + i * N);
    }

    // Gather the results back to the root process
    MPI_Gather(local_block.data(), block_size * N, MPI_DOUBLE,
               matrix.data(), block_size * N, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    // Ensure all processes finish before printing
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Matrix inversion complete. Inverted matrix:" << std::endl;
        print_matrix(matrix, N, N);
    }

    MPI_Finalize();
    return 0;
}

