import numpy as np


# Zig-zag matrix scanning

def zig_zag_vector(matrix):
    rows, cols = matrix.shape
    number_diag = rows + cols - 1
    min_dim, max_dim = (rows, cols) if rows < cols else (cols, rows)
    vector = []

    for diag in range(number_diag):
        if diag < min_dim:
            diag_len = diag + 1
        elif diag < max_dim:
            diag_len = min_dim
        else:
            diag_len = number_diag - diag

        for i in range(diag_len):
            if diag % 2 == 0:
                correct = 0 if diag < rows else rows - diag - 1  # vertical correction
                vector.append(matrix[diag - i + correct, i - correct])
            else:
                correct = 0 if diag < cols else cols - diag - 1  # horizontal correction
                vector.append(matrix[i - correct, diag - i + correct])

    return vector


def zig_zag_matrix(vector, rows, cols):
    number_diag = rows + cols - 1
    matrix = np.empty((rows, cols), dtype=np.int16)
    min_dim, max_dim = (rows, cols) if rows < cols else (cols, rows)

    j = 0
    for diag in range(number_diag):
        if diag < min_dim:
            diag_len = diag + 1
        elif diag < max_dim:
            diag_len = min_dim
        else:
            diag_len = number_diag - diag

        for i in range(diag_len):
            if diag % 2 == 0:
                correct = 0 if diag < rows else rows - diag - 1  # vertical correction
                matrix[diag - i + correct, i - correct] = vector[j]
            else:
                correct = 0 if diag < cols else cols - diag - 1  # horizontal correction
                matrix[i - correct, diag - i + correct] = vector[j]
            j += 1

    return matrix


def get_matrix_dimensions():
    while True:
        try:
            rows = int(input("Enter number of rows: "))
            cols = int(input("Enter the number of columns: "))
            print()

            if rows > 0 and cols > 0:
                return rows, cols
            else:
                print("The number of rows and columns must be greater than zero.")

        except ValueError:
            print('Invalid input. Enter whole numbers.')


if __name__ == '__main__':
    row, col = get_matrix_dimensions()
    source_matrix = np.random.randint(10, 100, size=(row, col))

    print('Source_matrix:\n', source_matrix, '\n')
    vector = zig_zag_vector(source_matrix)
    print('Matrix to vector:\n', vector, '\n')
    matrix = zig_zag_matrix(vector, row, col)
    print('Vector to matrix:\n', matrix)
