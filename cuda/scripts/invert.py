import numpy as np

# Define the matrix
matrix = np.array([
    [2.0, 2.0, 3.0, 4.0],
    [2.0, 2.0, 4.0, 5.0],
    [3.0, 4.0, 2.0, 6.0],
    [4.0, 5.0, 6.0, 2.0]
])

# Invert the matrix
try:
    inverse_matrix = np.linalg.inv(matrix)
    print("Inverse of the matrix:")
    print(inverse_matrix)
except np.linalg.LinAlgError:
    print("The matrix is singular and cannot be inverted.")

