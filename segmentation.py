import cv2
import numpy as np


def segmentation(image_path, R_value, G_value, B_value, pixel_count, global_mode, fill_mode, x_pos, y_pos):
    # print all values to console
    print(x_pos, y_pos, R_value, G_value, B_value, pixel_count, global_mode, fill_mode)
    if global_mode == 1:
        return global_segmentation(image_path, R_value, G_value, B_value, pixel_count)
    elif fill_mode == 1:
        img = cv2.imread(image_path)
        checked = np.zeros(img.shape)
        count = 0
        height, width, depth = img.shape
        fill_segmentation(img, checked, height, width, count, pixel_count, x_pos, y_pos)
    else:
        return global_segmentation(image_path, R_value, G_value, B_value, pixel_count)


def global_segmentation(image_path, R_val, G_val, B_val, pixel_count=250000):
    if pixel_count == 0:
        pixel_count = 250000
    tmp = cv2.imread(image_path)
    height, width, depth = tmp.shape
    output_image = np.zeros(tmp.shape)
    count = 0
    UP = 1.2
    LOW = 0.8

    R_val_UP = round(UP * R_val)
    R_val_LOW = round(LOW * R_val)
    G_val_UP = round(UP * G_val)
    G_val_LOW = round(LOW * G_val)
    B_val_UP = round(UP * B_val)
    B_val_LOW = round(LOW * B_val)

    for i in range(height):
        for j in range(width):
            if count < pixel_count:
                if R_val_LOW <= tmp[i, j, 2] <= R_val_UP \
                        and G_val_LOW <= tmp[i, j, 1] <= G_val_UP \
                        and B_val_LOW <= tmp[i, j, 0] <= B_val_UP:
                    output_image[i, j] = tmp[i, j]
                    count += 1
    return output_image


def fill_segmentation(image, checked, height, width, count, pixel_count, x, y):
    if count < pixel_count:

        if x + 1 < width and checked[y, x+1] == 0:
            fill_segmentation(image, checked, height, width, count, pixel_count, x + 1, y)
        if x - 1 < 0 and checked[y, x-1] == 0:
            fill_segmentation(image, checked, height, width, count, pixel_count, x - 1, y)
        if y + 1 < height and checked[y+1, x] == 0:
            fill_segmentation(image, checked, height, width, count, pixel_count, x, y + 1)
        if y - 1 < height and checked[y-1, x] == 0:
            fill_segmentation(image, checked, height, width, count, pixel_count, x, y - 1)
    return
