import numpy as np
from PIL import Image


# Upsampling image

def upsamp_channel(image, cx, cy):
    rows, cols = image.shape
    new_rows, new_cols = rows * cy, cols * cx

    upsamp_image = np.empty([new_rows, new_cols], dtype=np.uint8)

    for i in range(rows):
        for j in range(cols):
            upsamp_image[i * cy: (i + 1) * cy, j * cx: (j + 1) * cx] = image[i, j]

    return upsamp_image


def test_unsampling(name_image, cx, cy):
    try:
        with Image.open(name_image) as input_image:
            channels = np.array(input_image)

            first_channel = channels[:, :, 0]
            second_channel = channels[:, :, 1]
            third_channel = channels[:, :, 2]

            first_layer = upsamp_channel(first_channel, cx, cy)
            second_layer = upsamp_channel(second_channel, cx, cy)
            third_layer = upsamp_channel(third_channel, cx, cy)

            drop_image = np.stack([first_layer, second_layer, third_layer], axis=-1).astype(np.uint8)

            image = Image.fromarray(drop_image)
            name_save_image = name_image.rpartition('.')[0] + '_unsamp' + '.png'
            image.save(name_save_image)

            print(f"New size image: {image.size}")
            image.show()

    except Exception as error:
        print("Error: ", error)


if __name__ == '__main__':
    name_source_image = 'images_upsampling/killua.png'
    # name_source_image = 'images_upsampling/killua2.png'

    coeff_cx = 2 # horizontal expansion coefficient
    coeff_cy = 3 # vertical expansion coefficient

    test_unsampling(name_source_image, coeff_cx, coeff_cy)
