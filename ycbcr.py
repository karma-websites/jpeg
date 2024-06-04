import numpy as np
from PIL import Image


# Converting an RGB image to YCbCR color space

def rgb_to_ycbcr(image):
    if not isinstance(image, np.ndarray):
        raise TypeError("Input image must be a numpy array.")
    
    # Define the conversion matrix from RGB to YCbCr (BT.601 standard)
    conversion_matrix = np.array([
        [0.299,  0.587,  0.114],
        [-0.168736, -0.331264,  0.5],
        [0.5, -0.418688, -0.081312]
    ])
    
    # Offset for Cb and Cr components
    offset = np.array([0, 128, 128])
    
    # Reshape the image array to (height * width, 3) for matrix multiplication
    shape = image.shape
    flat_image = image.reshape((-1, 3))
    
    # Apply the matrix transformation
    ycbcr_flat = flat_image.dot(conversion_matrix.T) + offset
    
    # Reshape back to the original image shape
    ycbcr_image = ycbcr_flat.reshape(shape)

    return ycbcr_image.astype(np.uint8)


def ycbcr_to_rgb(image):
    if not isinstance(image, np.ndarray):
        raise TypeError("Input image must be a numpy array.")
    
    # Define the inverse conversion matrix from YCbCr to RGB (BT.601 standard)
    inverse_conversion_matrix = np.array([
        [1.0, 0.0, 1.402],
        [1.0, -0.344136, -0.714136],
        [1.0, 1.772, 0.0]
    ])
    
    # Offset for Cb and Cr components
    offset = np.array([0, 128, 128])
    
    # Reshape the image array to (height * width, 3) for matrix multiplication
    shape = image.shape
    flat_image = image.reshape((-1, 3)).astype(np.float32)
    
    # Remove the offsets from the Cb and Cr components
    flat_image -= offset
    
    # Apply the matrix transformation
    rgb_flat = flat_image.dot(inverse_conversion_matrix.T)
    
    # Reshape back to the original image shape
    rgb_image = rgb_flat.reshape(shape)

    return rgb_image.astype(np.uint8)


if __name__ == '__main__':
    try:
        # Load an RGB image using PIL
        name_source_image = 'images_ycbcr/lena.png'
        with Image.open(name_source_image) as input_image:
            rgb_image = np.array(input_image)

            # Convert RGB to YCbCr
            ycbcr_image = rgb_to_ycbcr(rgb_image)

            # Convert YCbCr image back to PIL image for saving or displaying
            ycbcr_image_pil = Image.fromarray(ycbcr_image)

            # Save the YCbCr image
            name_image = name_source_image.rpartition('.')[0] + '_ycbcr' + '.png'
            ycbcr_image_pil.save(name_image)

        name_source_image = 'images_ycbcr/lena_ycbcr.png'
        with Image.open(name_source_image) as input_image:
            ycbcr_image = np.array(input_image)

            # Convert YCbCr to RGB
            rgb_image = ycbcr_to_rgb(ycbcr_image)

            # Convert RGB image back to PIL image for saving or displaying
            rgb_image_pil = Image.fromarray(rgb_image)

            # Save the RGB image
            name_image = name_source_image.rpartition('_')[0] + '_rgb' + '.png'
            rgb_image_pil.save(name_image)

    except Exception as error:
        print('Error', error)
