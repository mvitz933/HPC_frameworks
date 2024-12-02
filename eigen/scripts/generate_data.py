import random

def generate_matrix(filename, rows, cols):
    with open(filename, 'w') as f:
        for i in range(rows):
            row = [str(random.randint(1, 100)) for _ in range(cols)]
            f.write(" ".join(row) + "\n")

if __name__ == "__main__":
    # Generate a 1000x1000 matrix
    generate_matrix("matrix_big.txt", 1000, 1000)
    print("Matrix generated and saved to matrix_big.txt")
