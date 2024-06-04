import numpy as np
from PIL import Image


def generate_single_color_image(size_image, color_image):
    image = Image.new('RGB', size_image, color_image)
    return image


def generate_discrete_image(size_image):
    width, height = size_image

    red_channel = np.random.randint(0, 256, size=(height, width))
    green_channel = np.random.randint(0, 256, size=(height, width))
    blue_channel = np.random.randint(0, 256, size=(height, width))

    noise_array = np.stack((red_channel, green_channel, blue_channel), axis=-1).astype(np.uint8)

    noise_image = Image.fromarray(noise_array)

    return noise_image


if __name__ == '__main__':
    color = (0, 0, 255)
    size = (1920, 1072)

    red_image = generate_single_color_image(size, color)
    red_image.save('images_generation/blue.png')

    discrete_image = generate_discrete_image(size)
    discrete_image.save('images_generation/discrete.png')


