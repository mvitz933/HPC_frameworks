Aggregation of tutorial material on CUDA programming

This repo contains CUDA-based implementations for solving numerical and machine learning problems. It leverages GPU acceleration to handle tasks like matrix operations, neural network training, and iterative solvers.  The neural network code, which constructs neural nets and trains them from scratch using CUDA, is adapted from the work of Daniel Warfield and Adrian Thoenig, see below:

https://www.youtube.com/watch?v=6StFanGtmvo

https://github.com/ThoenigAdrian/NeuralNetworksCudaTutorial/tree/main

https://iaee.substack.com/p/cuda-for-machine-learning-intuitively

https://github.com/DanielWarfield1/MLWritingAndResearch/blob/main/NNInCUDA.ipynb


Code Overview
Numerical Methods

    Matrix Operations:
        multiply_matrices.cu, invert_matrix.cu (and variations): Efficient GPU implementations for matrix multiplication and inversion.
        lu_decomposition.cu, lu_decomposition_shmem.cu: LU decomposition with and without shared memory optimizations.

    Iterative Solvers:
        solve_jacobi_naive.cu, solve_jacobi_shmem.cu, solve_jacobi_tiled_shmem.cu: Different versions of Jacobi solvers with optimizations for shared memory and tiling.

    Vector Operations:
        add_vectors.cu: Basic vector addition with CUDA.

Machine Learning

    Neural Networks:
        neural_network.cu, multiple_layers.cu: Implementations of multi-layer neural networks with backpropagation.
        linear_layer.cu, relu_activation.cu, sigmoid_activation.cu: Components like layers and activation functions.
        Training examples: training_with_linear.cu, training_with_relu.cu, training_with_sigmoid.cu.

    Cost Functions:
        bce_cost.cu: Binary Cross Entropy cost for classification tasks.

    Datasets:
        coordinates_dataset.cu: Dataset generation and management for neural network training.

Utilities and Demos

    Testing and Debugging:
        verify_matrix.cu, catch_exception.cu: Validation and exception handling examples.
        grid_dimensions.cu: Example to understand CUDA grid configurations.

    Scripts:
        compareMethods.py: Compare performance across different CUDA methods.
        invert.py: Numerical tools for matrix inversion.

Compile & Run

    Build:

make

Binaries go to build/bin/, and object files to build/obj/.

Run: Execute the compiled binaries:

    ./build/bin/<executable>

Outputs

    outputs/figures: Visualization files (e.g., plots).
    outputs/logs: Logs and debug output.
    outputs/numerical: Numerical results like matrix solutions.

Highlights

    Optimizations: Several files demonstrate optimizations like shared memory and tiling for efficient GPU usage.
    Modularity: Neural network components are modular and reusable.
    Flexibility: Code covers both general-purpose numerical methods and machine learning workflows.
