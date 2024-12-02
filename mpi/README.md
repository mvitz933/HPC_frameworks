MPI Project README

This repo has numerous MPI programs, originally from a 2017 HPC course at UVA, written for the Rivanna cluster. It’s been refactored for modern syntax where possible.
Code Overview
Numerical Methods

    Iterative Solvers: jacobi_mpi.cpp, jacobi_mpi_v2.cpp, gauss_seidel_mpi.cpp, red_black_mpi.cpp.
    Matrix Ops: mpi_matrix_inversion.cpp for parallel matrix inversion.

Monte Carlo and Random Walks

    π Estimation: mpi_pi_montecarlo.cpp, mpi_pi_trapezoid1.cpp, mpi_pi_trapezoid2.cpp.
    Random Walks: walkmpi.cpp, walkmpi_reduction.cpp.

Surface Analysis

    mpi_surface_maximum.cpp, mpi_surface_max_array.cpp: Grid-based max value finders.

Condition Satisfaction (SAT)

    sat1.cpp, sat2.cpp, sat3.cpp: MPI-based condition satisfaction problems.

Utilities and Demos

    hello_mpi.cpp, send_mpi.cpp: Simple MPI demos.
    random.cpp: Random number generation.

Compile & Run

Build:
```
make
```
Binaries go to build/bin/.

Run:

    mpirun -np 4 ./build/bin/<executable>

    Scripts: Use scripts/mpi_run.sh or submit_hello.slurm for automation.

Outputs

    outputs/figures: Visualizations.
    outputs/logs: Logs.
    outputs/numerical: Simulation results.
