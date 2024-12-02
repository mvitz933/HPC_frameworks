This repo has OpenMP-based code, originally from a 2017 HPC course at UVA. It’s been refactored for modern compilers and cleaner syntax.
Code Overview
Numerical Methods

    Decompositions: cholesky.cpp, lu.cpp, qr.cpp.
    Matrix Operations: invert.cpp, multiply.cpp, eigenvalues.cpp.
    Linear Solvers: solve.cpp, gaussseidel.cpp, jacobi.cpp.

Simulations

    Grid Solvers:
        solve_laplace2d.cpp, solve_poisson.cpp, solve_poisson_tiled.cpp.
        solve_heatedplate.cpp: Heat distribution solver.
        solve_red_black.cpp: Red-Black ordering for iterative solvers.

Compile & Run

    Build:

make

Binaries go to build/bin/, libraries to build/lib/, and object files to build/obj/.

Run:

    ./build/bin/<executable>

    Set Threads: Use scripts/setnumthreads.sh to configure the number of OpenMP threads.

Outputs

    outputs/figures: Plots and visualizations.
    outputs/logs: Logs and debug info.
    outputs/numerical: Simulation results.

Visualization Scripts

    MATLAB: plothistory.m, showplate.m for analyzing results.
    Python: mycontour.py for contour plots.

