import cv2
import numpy as np


def resize_keep_ratio(image: np.ndarray, new_size: int):
    h, w = image.shape[:2]
    if h > w:
        new_h = new_size
        ratio = h / new_size
        new_w = int(w / ratio)
    else:
        new_w = new_size
        ratio = w / new_size
        new_h = int(h / ratio)
    resized_image = cv2.resize(image, (new_w, new_h))
    return resized_image
