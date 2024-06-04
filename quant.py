import numpy as np
import dct

# Quantization

def get_quant_coeff(base_quant_coeff, q=50):
    s = max(1, min(100, q))
    q = 5000 / s if s < 50 else 200 - 2 * s
    quant_coeff = np.uint16((q * base_quant_coeff + 50) / 100)
    quant_coeff[quant_coeff == 0] = 1
    return quant_coeff


def quantize(dct_matrix, quant_coeff):
    return np.int16(np.divide(dct_matrix, quant_coeff).round())


def dequantize(quant_matrix, quant_coeff):
    return np.multiply(quant_matrix, quant_coeff)


base_quant_coeff_y = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])


base_quant_coeff_cbcr = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 47, 99, 99, 99, 99],
    [47, 66, 24, 47, 99, 99, 99, 99],
    [17, 18, 24, 47, 99, 99, 99, 99],
    [17, 18, 24, 47, 99, 99, 99, 99],
    [17, 18, 24, 47, 99, 99, 99, 99],
    [17, 18, 24, 47, 99, 99, 99, 99]
])


if __name__ == '__main__':
    # array = np.random.randint(0, 256, size=(8, 8), dtype=np.uint8)

    uint8_matrix = np.array([
        [52, 55, 61, 66, 70, 61, 64, 73],
        [63, 59, 55, 90, 109, 85, 69, 72],
        [62, 59, 68, 113, 144, 104, 66, 73],
        [63, 58, 71, 122, 154, 106, 70, 69],
        [67, 61, 68, 104, 126, 88, 68, 70],
        [79, 65, 60, 70, 77, 68, 58, 75],
        [85, 71, 64, 59, 55, 61, 65, 83],
        [87, 79, 69, 68, 65, 76, 78, 94]
    ])

    int8_matrix = dct.range_change_int8(uint8_matrix)

    dct_matr = dct.dct(int8_matrix)

    quant_coeff_y = get_quant_coeff(base_quant_coeff_y, 75)

    quantized_matrix = quantize(dct_matr, quant_coeff_y)
    dequantized_matrix = dequantize(quantized_matrix, quant_coeff_y)

    print('Quant matrix Y components for Q = 75:\n', quant_coeff_y, '\n')
    print('Dct matrix:\n', np.int16(dct_matr), '\n')
    print('Quantized matrix:\n', quantized_matrix, '\n')
    print('Dequantized matrix:\n', dequantized_matrix, '\n')
