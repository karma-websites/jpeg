import numpy as np
from PIL import Image


# Downsampling image

def downsamp_channel(channel, cx, cy, method):
    if cx == 1 and cy == 1:
        return channel

    rows, cols = channel.shape
    new_rows, new_cols = rows // cx, cols // cy

    # Downsampling the image channel by removing rows and columns
    if method == 'drop':
        return channel[::cx, ::cy]

    # Downsampling the image channel by replacing the block with a pixel
    # with the average color value of the block
    elif method == 'average':
        average_image = np.empty([new_rows, new_cols], dtype=np.uint8)
        for i in range(new_rows):
            for j in range(new_cols):
                block = channel[i * cx: (i + 1) * cx, j * cy: (j + 1) * cy]
                average_pixel = np.uint8(np.mean(block))
                average_image[i, j] = average_pixel
        return average_image

    # Downsampling the image channel by replacing the block with a pixel
    # closest in value to the average
    elif method == 'nearest':
        nearest_image = np.empty([new_rows, new_cols], dtype=np.uint8)
        for i in range(new_rows):
            for j in range(new_cols):
                block = channel[i * cx:(i + 1) * cx, j * cy:(j + 1) * cy]
                average_pixel = np.uint8(np.mean(block))
                nearest_pixel = block.flat[np.abs(block - average_pixel).argmin()]
                nearest_image[i, j] = nearest_pixel
        return nearest_image

    else:
        raise ValueError('Wrong way to downsample')


def test_downsampling(name_image, cx, cy, method):
    try:
        with Image.open(name_image) as input_image:
            channels = np.array(input_image)

            first_channel = channels[:, :, 0]
            second_channel = channels[:, :, 1]
            third_channel = channels[:, :, 2]

            first_layer = downsamp_channel(first_channel, cx, cy, method)
            second_layer = downsamp_channel(second_channel, cx, cy, method)
            third_layer = downsamp_channel(third_channel, cx, cy, method)

            drop_image = np.stack([first_layer, second_layer, third_layer], axis=-1)

            image = Image.fromarray(drop_image)
            name_save_image = name_image.rpartition('.')[0] + '_' + method + 'samp' + '.png'
            image.save(name_save_image)

            print(f'{method} downsampling')
            print(f"New size image: {image.size}")
            image.show()

    except Exception as error:
        print("Error: ", error)


if __name__ == '__main__':
    name_source_image = 'images_downsampling/killua.bmp'

    coeff_cx = 5  # horizontal compression coefficient
    coeff_cy = 4  # vertical compression coefficient

    test_downsampling(name_source_image, coeff_cx, coeff_cy, 'drop')
    test_downsampling(name_source_image, coeff_cx, coeff_cy, 'average')
    test_downsampling(name_source_image, coeff_cx, coeff_cy, 'nearest')
