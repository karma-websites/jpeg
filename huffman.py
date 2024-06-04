import dahuffman


def compress_huffman(name_input_file):
    with open(name_input_file, 'rb') as input_file:
        byte_data = input_file.read()
        codec = dahuffman.HuffmanCodec.from_data(byte_data)
        encode_data = codec.encode(byte_data)

    name_output_file = name_input_file.rpartition('.')[0] + '.huff'
    with open(name_output_file, "wb") as output_file:
        output_file.write(encode_data)

    return codec


def decompress_huffman(name_file, codec):
    with open(name_file, "rb") as input_file:
        byte_data = input_file.read()
        decode_data = codec.decode(byte_data)

    name_output_file = name_file.rpartition('.')[0] + '_inverse.myjpg'
    with open(name_output_file, "wb") as output_file:
        output_file.write(decode_data)


if __name__ == '__main__':
    name_myjpeg_file = 'images_huffman/killua.myjpg'
    name_huffman_file = 'images_huffman/killua.huff'

    table = compress_huffman(name_myjpeg_file)

    print(table.print_code_table())

    decompress_huffman(name_huffman_file, table)
