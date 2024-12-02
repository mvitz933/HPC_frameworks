Project Overview

This repository contains implementations using various parallel computing frameworks and numerical libraries, including CUDA, Eigen, MPI, OpenACC, and OpenMP. A shared directory structure and compilation process are used across all frameworks to maintain consistency and modularity. Additionally, a common approach to handling tasks was applied wherever possible to ensure uniformity across implementations.

Directory Structure
```
framework/
├── build         # Compiled binaries and build artifacts
├── include       # Header files
├── main          # Main entry points for the program
├── Makefile      # Build system configuration
├── outputs       # Output data, logs, or results
├── README.md     # Framework-specific documentation
├── scripts       # Helper scripts for running or automating tasks
└── src           # Source files
```
This directory structure is repeated for each of the following frameworks:

    cuda/
    eigen/
    mpi/
    openacc/
    openmp/

(Strictly speaking, Eigen is not a framework, but it is a very useful library to compare against.)
(MPI has not been fully factorized into src and main code, 
because most of the logic is contained in the main program anyway. This is a TODO.)


There is also a data/ directory containing shared input data, like images and matrices.

Compilation and Execution
The process for compiling and running executables is the same for all frameworks:

Navigate to the Desired Framework:
```
cd <framework-name>  # Replace <framework-name> with cuda, eigen, mpi, etc.
```
Compile the Program: Use the Makefile to build the executable:
```
make
```
Run the Program: Execute the compiled binary from the build/ directory:
```
    ./build/your_program
```
Access Outputs: Check the outputs/ directory for results, logs, or generated files.

Design Philosophy

A common approach was applied across all frameworks wherever possible. This means:

    Consistent directory structure for easier navigation.
    Shared compilation process using Makefiles.
    Standardized handling of inputs and outputs, such as images and matrices stored in the data/ directory.

This is supposed to make it easy to navigate and understand the workflows.

Planned improvements and general TODO list (not in order of priority):

    1. Uniform testing and comparison scripts:
        Develop common testing and benchmarking scripts to compare the performance of executables across frameworks.
        Generate detailed performance metrics, such as runtime, memory usage, and scalability.

    2. Utilities for generating of data:
        Create utility scripts or programs to pull standard datasets of generate synthetic for testing and benchmarking.
        Include better support for configurable parameters (e.g., matrix size, data distribution types).

    3. Matrix inversion debugging:
        My matrix inversion programs don't work in cuda 

    4. Use CMake :
        Go from individual Makefiles to a unified CMake build system.
        Simplify dependency management and cross-platform compatibility.

    5. More Algorithms:
        Expand to include extra algorithms:
            Matrix factorizations (SVD).
            Optimization techniques (gradient descent, conjugate gradient).
            Signal processing methods (Fourier transforms, wavelet transforms).
            Graph algorithms (shortest paths, connected components).
            More machine learning and data engineering.

    6. Factorize the MPI code and modernize some older code in it

