This repo uses OpenACC to solve numerical problems, getting GPU acceleration for tasks like matrix decompositions, solving linear systems, and iterative methods. The implementation uses header-only methods with templates, so the src folder only contains placeholder .txt files for reference.
Code Overview
Matrix Operations

    Matrix Multiplication: multiply.hpp template-based matrix multiplication.
    Matrix Inversion: invert.hpp for solving inverse matrices.
    Eigenvalues: eigenvalues.hpp for calculating eigenvalues of matrices.

Matrix Decompositions

    Cholesky: cholesky.hpp for Cholesky decomposition.
    LU: lu.hpp for LU decomposition.
    QR: qr.hpp for QR decomposition.

Linear Solvers

    Jacobi and Gauss-Seidel: jacobi.hpp, gaussseidel.hpp implement iterative solvers.
    Poisson and Laplace Solvers: methods for solving 2D Poisson and Laplace equations (solve.hpp).

Compile & Run

    Build:

make

Binaries are in build/bin/ and object files in build/obj/.

Run: Execute the compiled binaries:

    ./build/bin/<executable>

Outputs

    outputs/figures: visualizations of results.
    outputs/logs: Execution logs and debug 
    outputs/numerical: Numerical outputs, like computed matrices or iterative solver outputs.

Scripts

    cuda_runtime.cpp: Utility for getting CUDA runtime info
