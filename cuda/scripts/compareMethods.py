import numpy as np

def read_matrix_from_file(file_path):
    """Reads a matrix from a file."""
    try:
        with open(file_path, 'r') as f:
            # Read the matrix, each line contains comma-separated values
            matrix = np.array([list(map(float, line.strip().split(','))) for line in f])
        return matrix
    except Exception as e:
        print(f"Error reading matrix from {file_path}: {e}")
        return None

def compare_matrices(mat1, mat2, tolerance=1e-6):
    """Compares two matrices with a given tolerance."""
    if mat1.shape != mat2.shape:
        print("Matrices have different dimensions and cannot be compared.")
        return False
    return np.allclose(mat1, mat2, atol=tolerance)

def main():
    # File paths
    matrix_file = "randomMatrix_50.txt"
    inv_file = "inv.txt"

    # Read the matrix from file
    matrix = read_matrix_from_file(matrix_file)
    if matrix is None:
        return

    # Calculate the inverse using numpy
    try:
        matrix_inverse = np.linalg.inv(matrix)
        print("Calculated inverse using NumPy:")
        print(matrix_inverse)
    except np.linalg.LinAlgError:
        print("Matrix is singular and cannot be inverted.")
        return

    # Read the provided inverse matrix from inv.txt
    provided_inverse = read_matrix_from_file(inv_file)
    if provided_inverse is None:
        return

    # Compare the calculated inverse with the provided inverse
    if compare_matrices(matrix_inverse, provided_inverse):
        print("The calculated inverse matches the provided inverse.")
    else:
        print("The calculated inverse does NOT match the provided inverse.")

if __name__ == "__main__":
    main()

