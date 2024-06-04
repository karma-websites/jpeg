import numpy as np

# Dct conversion

def get_coeff(n):
    if n == 0:
        coeff = 1 / np.sqrt(2)
    else:
        coeff = 1
    return coeff


def dct(matrix):
    dct_matrix = np.empty([8, 8], dtype=np.float64)
    for u in range(8):
        for v in range(8):
            dct_matrix[u, v] = dct_uv(matrix, u, v)
    return dct_matrix


def dct_uv(matrix, u, v):
    result_uv = 0
    k1 = np.pi / 16 * u
    k2 = np.pi / 16 * v

    for i in range(8):
        for j in range(8):
            first_arg = (2 * i + 1) * k1
            second_arg = (2 * j + 1) * k2
            result_uv += matrix[i, j] * np.cos(first_arg) * np.cos(second_arg)

    cu = get_coeff(u)
    cv = get_coeff(v)
    result_uv *= 1/4 * cu * cv

    return result_uv


def idct(dct_matrix):
    matrix = np.empty([8, 8], dtype=np.float64)
    for i in range(8):
        for j in range(8):
            matrix[i, j] = idct_ij(dct_matrix, i, j)
    return matrix


def idct_ij(dct_matrix, i, j):
    result_ij = 0
    k1 = (2 * i + 1) * np.pi / 16
    k2 = (2 * j + 1) * np.pi / 16

    for u in range(8):
        for v in range(8):
            first_arg = k1 * u
            second_arg = k2 * v
            cu = get_coeff(u)
            cv = get_coeff(v)
            result_ij += dct_matrix[u, v] * cu * cv * np.cos(first_arg) * np.cos(second_arg)

    result_ij *= 1 / 4

    return result_ij


def range_change_int8(matrix):
    return np.int8(matrix.clip(0, 255) - 128)


def range_change_uint8(matrix):
    return np.uint8(matrix.clip(-128, 127) + 128)


def get_dct_coeff():
    dct_coeff = np.empty([8, 8], dtype=np.float64)

    for j in range(8):
            dct_coeff[0, j] = 1 / np.sqrt(8)

    for i in range(1, 8):
        for j in range(8):
            dct_coeff[i, j] = 1 / 2 * np.cos((2 * j + 1) * i * np.pi / 16)

    return dct_coeff


def dct2(matrix, dct_coeff):
    return dct_coeff.dot(matrix).dot(dct_coeff.T)


def idct2(dct_matrix, dct_coeff):
    return dct_coeff.T.dot(dct_matrix).dot(dct_coeff)


def test_dct(uint8_matrix, int8_matrix):
    dct_matrix = dct(int8_matrix)
    print('DCT matrix:\n', dct_matrix, '\n')

    idct_matrix = np.int8(idct(dct_matrix).round())
    print('IDCT matrix:\n', idct_matrix, '\n')

    idct_matrix = range_change_uint8(idct_matrix)
    print('Source range IDCT matrix:\n', idct_matrix, '\n')

    print('Comparison of the source and IDCT matrices:\n', uint8_matrix == idct_matrix)


def test_dct2(uint8_matrix, int8_matrix):
    dct_coeff = get_dct_coeff()

    dct_matrix = dct2(int8_matrix, dct_coeff)
    print('DCT2 matrix:\n', dct_matrix, '\n')

    idct_matrix = np.int8(idct2(dct_matrix, dct_coeff).round())
    print('IDCT2 matrix:\n', idct_matrix, '\n')

    idct_matrix = range_change_uint8(idct_matrix)
    print('Source range IDCT2 matrix:\n', idct_matrix, '\n')

    print('Comparison of the source and IDCT2 matrices:\n', uint8_matrix == idct_matrix)


if __name__ == '__main__':
    # array = np.random.randint(0, 256, size=(8, 8), dtype=np.uint8)

    source_matrix = np.array([
        [52, 55, 61, 66, 70, 61, 64, 73],
        [63, 59, 55, 90, 109, 85, 69, 72],
        [62, 59, 68, 113, 144, 104, 66, 73],
        [63, 58, 71, 122, 154, 106, 70, 69],
        [67, 61, 68, 104, 126, 88, 68, 70],
        [79, 65, 60, 70, 77, 68, 58, 75],
        [85, 71, 64, 59, 55, 61, 65, 83],
        [87, 79, 69, 68, 65, 76, 78, 94]
    ])

    print('Source matrix:\n', source_matrix, '\n')
    range_matrix = range_change_int8(source_matrix)
    print('New range matrix:\n', range_matrix, '\n\n')

    test_dct(source_matrix, range_matrix)
    print('\n')
    test_dct2(source_matrix, range_matrix)


