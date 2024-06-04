import numpy as np
from PIL import Image
import os.path

import ycbcr2
import dct
import quant
import zig_zag
import rle
import downsampling
import upsampling
import huffman


def get_quant_matrix(name_channel, quality_factor):
    if name_channel == 'y':
        return quant.get_quant_coeff(quant.base_quant_coeff_y, quality_factor)
    return quant.get_quant_coeff(quant.base_quant_coeff_cbcr, quality_factor)


def compress_channel(channel_image, name_channel, quality_factor):
    rows, cols = channel_image.shape
    quant_channel = np.empty([rows, cols], dtype=np.int16)
    comp_channel = []

    dct_coeff = dct.get_dct_coeff()
    quant_coeff = get_quant_matrix(name_channel, quality_factor)

    for i in range(0, rows, 8):
        for j in range(0, cols, 8):
            block = channel_image[i:i + 8, j:j + 8]
            change_block = dct.range_change_int8(block)
            dct_block = dct.dct2(change_block, dct_coeff)
            quant_block = quant.quantize(dct_block, quant_coeff)
            quant_channel[i:i + 8, j:j + 8] = quant_block
            vector_block = zig_zag.zig_zag_vector(quant_block)
            rle_block = rle.rle_encode(vector_block)
            comp_channel.extend(rle_block)

    comp_channel = np.asarray(comp_channel, dtype=np.int16)
    return quant_channel, comp_channel


def ijpg(vector_image, image_size, name_channel, quality_factor):
    if name_channel != 'y':
        rows, cols = image_size
        rows = rows // 2
        cols = cols // 2
    else:
        rows, cols = image_size

    ijpg_channel = np.empty([rows, cols], dtype=np.uint8)

    dct_coeff = dct.get_dct_coeff()
    quant_coeff = get_quant_matrix(name_channel, quality_factor)

    decoding = rle.rle_decode(vector_image)
    k = 0
    for i in range(0, rows, 8):
        for j in range(0, cols, 8):
            vector = decoding[k: k + 64]
            block = zig_zag.zig_zag_matrix(vector, 8, 8)
            dequant_block = quant.dequantize(block, quant_coeff)
            idct_block = dct.idct2(dequant_block, dct_coeff)
            change_block = dct.range_change_uint8(idct_block)
            ijpg_channel[i:i+8, j:j+8] = change_block
            k += 64

    return ijpg_channel


def write_channels(name_compress_file, quality_factor, image_size, comp_channel1, comp_channel2, comp_channel3):
    with open(name_compress_file, 'wb') as file:
        file.write(quality_factor.to_bytes(1, byteorder='little', signed=False))

        rows, cols = image_size
        file.write(rows.to_bytes(2, byteorder='little', signed=False))
        file.write(cols.to_bytes(2, byteorder='little', signed=False))

        len_channel1 = len(comp_channel1)
        file.write(len_channel1.to_bytes(4, byteorder='little', signed=False))
        len_channel2 = len(comp_channel2)
        file.write(len_channel2.to_bytes(4, byteorder='little', signed=False))
        len_channel3 = len(comp_channel3)
        file.write(len_channel3.to_bytes(4, byteorder='little', signed=False))

        file.write(comp_channel1.tobytes())
        file.write(comp_channel2.tobytes())
        file.write(comp_channel3.tobytes())


def read_channels(name_comp_image):
    with open(name_comp_image, 'rb') as file:
        byte_data = file.read(1)
        quality_factor = np.frombuffer(byte_data, dtype=np.uint8).item()

        byte_data = file.read(2)
        rows = np.frombuffer(byte_data, dtype=np.uint16)
        byte_data = file.read(2)
        cols = np.frombuffer(byte_data, dtype=np.uint16)
        image_size = rows.item(), cols.item()

        byte_data = file.read(12)
        len_channels = np.frombuffer(byte_data, dtype=np.uint32)

        byte_data = file.read()
        channels = np.frombuffer(byte_data, dtype=np.int16)

        return quality_factor, image_size, len_channels, channels


def save_quant_image(name_quant_image, quant_channel1, quant_channel2, quant_channel3):
    quant_image = np.stack([quant_channel1, quant_channel2, quant_channel3], axis=-1)
    quant_image_pil = quant_image.astype(np.uint8)
    image = Image.fromarray(quant_image_pil)
    image.save(name_quant_image)


def compress_image(name_source_image, qual_factor):
    with Image.open(name_source_image) as input_image:
        channels = np.array(input_image)

        y, cb, cr = ycbcr2.rgb_to_ycbcr(channels)
        quant_channel1, comp_channel1 = compress_channel(y, 'y', qual_factor)
        compress_cb = downsampling.downsamp_channel(cb, 2, 2, 'average')
        quant_channel2, comp_channel2 = compress_channel(compress_cb, 'cb', qual_factor)
        compress_cr = downsampling.downsamp_channel(cr, 2, 2, 'average')
        quant_channel3, comp_channel3 = compress_channel(compress_cr, 'cr', qual_factor)

        name_compress_file = name_source_image.rpartition('.')[0] + '.myjpg'
        image_size = y.shape
        write_channels(name_compress_file, qual_factor, image_size, comp_channel1, comp_channel2, comp_channel3)

        # Save quant image
        name_quant_image = name_source_image.rpartition('.')[0] + '_quant' + '.png'
        compress_y = downsampling.downsamp_channel(y, 2, 2, 'average')
        quant_ch1, comp_ch1 = compress_channel(compress_y, 'y', qual_factor)
        save_quant_image(name_quant_image, quant_ch1, quant_channel2, quant_channel3)


def decompress_image(name_compress_file):
    quality_factor, image_size, len_channels, channels = read_channels(name_compress_file)

    vector_image1 = channels[0:len_channels[0]]
    len_channel12 = len_channels[0] + len_channels[1]
    vector_image2 = channels[len_channels[0]:len_channel12]
    len_channel123 = len_channel12 + len_channels[2]
    vector_image3 = channels[len_channel12:len_channel123]

    first_layer = ijpg(vector_image1, image_size, 'y', quality_factor)
    second_layer = ijpg(vector_image2, image_size, 'cb', quality_factor)
    decomp_second_layer = upsampling.upsamp_channel(second_layer, 2, 2)
    third_layer = ijpg(vector_image3, image_size, 'cr', quality_factor)
    decomp_third_layer = upsampling.upsamp_channel(third_layer, 2, 2)

    rgb_image = ycbcr2.ycbcr_to_rgb(first_layer, decomp_second_layer, decomp_third_layer)

    image = Image.fromarray(rgb_image)
    name_image = name_compress_file.rpartition('.')[0] + '_ijpg' + '.png'
    image.save(name_image)


def test_jpg(name_orig_image, name_compress_image, name_huff_image):
    for i in range(0, 4):
        print(name_orig_image[i], '\n')
        for j in range(100, 9, -10):
            print('Factor Q: ', j)
            compress_image(name_orig_image[i], j)
            print('File size in bytes after rle: ', os.path.getsize(name_compress_image[i]))
            huffman.compress_huffman(name_compress_image[i])
            print('File size in bytes after huffman: ', os.path.getsize(name_huff_image[i]), '\n')
        print()


if __name__ == '__main__':
    path_orig_image = ['images_tests/killua.bmp', 'images_tests/gradient.bmp', 'images_tests/blue.bmp', 'images_tests/discrete.bmp']
    path_compress_image = ['images_tests/killua.myjpg', 'images_tests/gradient.myjpg', 'images_tests/blue.myjpg', 'images_tests/discrete.myjpg']
    path_huff_image = ['images_tests/killua.huff', 'images_tests/gradient.huff', 'images_tests/blue.huff', 'images_tests/discrete.huff']

    # test_jpg(orig_image, compress_image, huff_image)

    quality_factor = 50
    index = 0;

    compress_image(path_orig_image[index], quality_factor)
    print('File size in bytes after rle: ', os.path.getsize(path_compress_image[index]))

    huffman.compress_huffman(path_compress_image[index])
    print('File size in bytes after huffman: ', os.path.getsize(path_huff_image[index]), '\n')

    decompress_image(path_compress_image[index])
