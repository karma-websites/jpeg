from PIL import Image


def save_image(name_source_image, name_save_image):
    try:
        with Image.open(name_source_image) as input_image:
            input_image.save(name_save_image)

    except FileNotFoundError:
        print(f'File {name_save_image} not found')


if __name__ == '__main__':
    name_source_image = 'images_jpg/killua.jpg'
    format_image = '.bmp'
    name_save_image = name_source_image.rpartition('.')[0] + format_image
    save_image(name_source_image, name_save_image)
