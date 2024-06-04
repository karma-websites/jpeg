import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt


# Converting an RGB image to YCbCR color space

def rgb_to_ycbcr(rgb_image):
    y = 0.299 * rgb_image[:, :, 0] + 0.587 * rgb_image[:, :, 1] + 0.114 * rgb_image[:, :, 2]
    cb = 128.0 - 0.168736 * rgb_image[:, :, 0] - 0.331264 * rgb_image[:, :, 1] + 0.5 * rgb_image[:, :, 2]
    cr = 128.0 + 0.5 * rgb_image[:, :, 0] - 0.418688 * rgb_image[:, :, 1] - 0.081312 * rgb_image[:, :, 2]
    y = np.uint8(y.clip(0, 255))
    cb = np.uint8(cb.clip(0, 255))
    cr = np.uint8(cr.clip(0, 255))
    return y, cb, cr


def ycbcr_to_rgb(y, cb, cr):
    r = y + 1.402 * (cr - 128.0)
    g = y - 0.344136 * (cb - 128.0) - 0.714136 * (cr - 128.0)
    b = y + 1.772 * (cb - 128.0)
    merge_image = np.stack([r, g, b], axis=-1)
    return np.uint8(merge_image.clip(0, 255))


if __name__ == '__main__':
    try:
        image = cv2.imread('images_ycbcr/killua.png')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        y, cb, cr = rgb_to_ycbcr(image)

        ycbcr_image = cv2.merge([y, cb, cr])
        ycbcr_image_pil = Image.fromarray(ycbcr_image)
        ycbcr_image_pil.save('images_ycbcr/ycbcr.png')

        rgb_image = ycbcr_to_rgb(y, cb, cr)
        rgb_image_pil = Image.fromarray(rgb_image)
        rgb_image_pil.save('images_ycbcr/rgb.png')

        y_image = cv2.merge([y, y, y])
        y_image_pil = Image.fromarray(y_image)
        y_image_pil.save('images_ycbcr/y_component.png')

        cb_image = cv2.merge([cb, cb, cb])
        cb_image_pil = Image.fromarray(cb_image)
        cb_image_pil.save('images_ycbcr/cb_component.png')

        cr_image = cv2.merge([cr, cr, cr])
        cr_image_pil = Image.fromarray(cr_image)
        cr_image_pil.save('images_ycbcr/cr_component.png')

        fig, axes = plt.subplots(2, 3, figsize=(12, 6))
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original')
        axes[0, 1].imshow(y_image)
        axes[0, 1].set_title('Y component')
        axes[0, 2].imshow(cb_image)
        axes[0, 2].set_title('Cb component')
        axes[1, 0].imshow(cr_image)
        axes[1, 0].set_title('Cr component')
        axes[1, 1].imshow(ycbcr_image)
        axes[1, 1].set_title('YCbCr image')
        axes[1, 2].imshow(rgb_image)
        axes[1, 2].set_title('RGB image')
        for a in axes:
            for b in a:
                b.axis('off')
        fig.suptitle('Converting an RGB image to YCbCR color space')
        plt.show()

    except Exception as error:
        print('Error', error)
