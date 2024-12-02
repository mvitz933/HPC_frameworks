import numpy as np
from scipy.linalg import lu

# Define the matrix A
A = np.array([
    [2, -1, 1],
    [3, 3, 9],
    [3, 3, 5]
])

# Perform LU decomposition
P, L, U = lu(A)

# Print the matrices
print("Matrix A:")
print(A)

print("\nPermutation matrix P:")
print(P)

print("\nLower triangular matrix L:")
print(L)

print("\nUpper triangular matrix U:")
print(U)

