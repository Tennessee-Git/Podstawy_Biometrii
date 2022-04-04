import numpy as np
import cv2
import math
import matplotlib.pyplot as plt


def filter_image(image_path, pixel_size, filter_type):
    pixel_size = int(pixel_size)
    if filter_type == 'Pixelization':
        return pixelization(image_path, pixel_size)
    elif filter_type == 'Median Filter':
        return median_filter(image_path, pixel_size)
    elif filter_type == 'Kuwahara Filter':
        return kuwahara_filter(image_path, pixel_size)
    elif filter_type == 'Gaussian Filter':
        return gaussian_filter(image_path, pixel_size)
    elif filter_type == 'Sobel Filter':
        sobel_filter(image_path, pixel_size)


def dnorm(x, mu, sd):
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)


def gaussian_kernel(size, sigma=1):
    kernel_1D = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        kernel_1D[i] = dnorm(kernel_1D[i], 0, sigma)
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)

    kernel_2D *= 1.0 / kernel_2D.max()

    return kernel_2D


def gaussian_filter(image_path, pixel_size):
    image = cv2.imread(image_path)
    kernel = gaussian_kernel(pixel_size, sigma=math.sqrt(pixel_size))
    """
    [[0.44932896 0.60653066 0.67032005 0.60653066 0.44932896]
     [0.60653066 0.81873075 0.90483742 0.81873075 0.60653066]
     [0.67032005 0.90483742 1.         0.90483742 0.67032005]
     [0.60653066 0.81873075 0.90483742 0.81873075 0.60653066]
     [0.44932896 0.60653066 0.67032005 0.60653066 0.44932896]]
    """
    return convolution(image, kernel)


def convolution(image, kernel):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape

    output = np.zeros(image.shape)

    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)

    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))

    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image

    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
            output[row, col] /= kernel.shape[0] * kernel.shape[1]

    return output


def sobel_filter(image_path, pixel_size):
    filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    image = cv2.imread(image_path)
    image = gaussian_filter(image_path, 9)
    return sobel_edge_detection(image, filter)

    pass


def sobel_edge_detection(image, filter):
    new_image_x = convolution(image, filter)
    new_image_y = convolution(image, np.flip(filter.T, axis=0))
    gradient_magnitude = np.sqrt(np.square(new_image_x) + np.square(new_image_y))
    gradient_magnitude *= 255.0 / gradient_magnitude.max()

    plt.imshow(gradient_magnitude, cmap='gray')
    plt.title("Gradient Magnitude")
    plt.show()

    return new_image_x



def kuwahara_filter(image_path, pixel_size):
    pass


def pixelization(image_path, pixel_size):
    tmp = cv2.imread(image_path)
    blue, green, red = cv2.split(tmp)
    height, width = blue.shape
    for x in range(0, width, pixel_size):
        for y in range(0, height, pixel_size):
            endX = x + pixel_size
            endY = y + pixel_size
            if endX > width:
                endX = width
            if endY > height:
                endY = height
            blue[x:endX, y:endY] = blue[x:x + pixel_size, y: y + pixel_size].mean()
            green[x:endX, y:endY] = green[x:x + pixel_size, y: y + pixel_size].mean()
            red[x:endX, y:endY] = red[x:x + pixel_size, y: y + pixel_size].mean()
    output = cv2.merge((blue, green, red))
    return output


def median_filter(image_path, pixel_size):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    temp = []
    indexer = pixel_size // 2
    window = [  # offsety pikseli
        (i, j)
        for i in range(-indexer, pixel_size - indexer)
        for j in range(-indexer, pixel_size - indexer)
    ]
    index = len(window) // 2
    for i in range(len(image)):
        for j in range(len(image[0])):
            image[i][j] = sorted(
                0 if (  # 0 jeżeli i + offset lub j + offset wychodza poza obraz
                        min(i + a, j + b) < 0
                        or len(image) <= i + a
                        or len(image[0]) <= j + b
                ) else image[i + a][j + b]
                for a, b in window
            )[index]
    return image
