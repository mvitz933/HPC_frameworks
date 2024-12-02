This repo uses the Eigen C++ library for linear algebra, matrix decompositions, and numerical computations. The code is modular and demonstrates key algorithms for solving systems, decompositions, and matrix operations.
Code Overview
Matrix Decompositions

    Cholesky: cholesky.cpp, cholesky_decompose.cpp.
    LU: lu.cpp, lu_decompose.cpp.
    QR: qr.cpp, qr_decompose.cpp.

Matrix Operations

    Inversion: invert.cpp, matrix_invert.cpp.
    Multiplication: multiply.cpp, matrix_multiply.cpp.
    Eigenvalues: eigenvalues.cpp, matrix_eigenvalues.cpp.

Linear Solvers

    General Solver: solve.cpp, linear_solve.cpp.

Compile & Run

    Build:

make

Binaries go to build/bin/, libraries to build/lib/, and object files to build/obj/.

Run:

    ./build/bin/<executable>

Outputs

    outputs/: Stores results such as numerical outputs or logs.

Utilities

    Data Generation:
        Use scripts/generate_data.py to create synthetic input data for testing matrix operations and solvers.


