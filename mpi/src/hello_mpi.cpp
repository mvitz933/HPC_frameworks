#include <mpi.h>
#include <iostream>

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get the rank of the current process
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Get the total number of processes
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Print a message from this process
    std::cout << "Hello from process " << rank << " out of " << size << " processes!" << std::endl;

    // Finalize MPI
    MPI_Finalize();

    return 0;
}

