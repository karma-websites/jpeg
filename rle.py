# Rle

def rle_encode_jpg(data):
    encoding = []
    count = 0
    for value in data:
        if value == 0:
            count += 1
        else:
            encoding.append(count)
            encoding.append(value)
            count = 0
    encoding.append('eof')
    return encoding


def rle_encode(data):
    encoding = []
    special_byte = 255
    count = 1
    i = 0

    while i < len(data):
        while i < len(data) - 1 and data[i] == data[i + 1] == 0:
            count += 1
            i += 1
        if count > 1:
            encoding.append(special_byte)
            encoding.append(count)
        else:
            if data[i] == special_byte:
                encoding.append(special_byte)
                encoding.append(special_byte)
            else:
                encoding.append(data[i])
        count = 1
        i += 1

    return encoding


def rle_decode(data):
    decoding = []
    special_byte = 255
    i = 0

    while i < len(data):
        if data[i] == special_byte:
            if data[i + 1] == special_byte:
                decoding.append(special_byte)
            else:
                for j in range(data[i + 1]):
                    decoding.append(0)
            i += 2
        else:
            decoding.append(data[i])
            i += 1

    return decoding


if __name__ == '__main__':
    quantized_coeffs = [255, 6, 0, 0, 8, 0, 4, -2, 0, 0, 0, -4, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    encoded_sequence = rle_encode(quantized_coeffs)
    decoded_sequence = rle_decode(encoded_sequence)

    print('Quantized_coeffs:\n', quantized_coeffs)
    print('Encoded sequence:\n', encoded_sequence)
    print('Decoded sequence:\n', decoded_sequence)
