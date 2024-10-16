import cv2
import numpy as np
import matplotlib.pyplot as plt

def local_histogram_equalization(image, window_size):
    """
    Funcion para mejorar el contraste en zonas específicas de la imagen.
    """
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    border_size = window_size // 2
    img_bordered = cv2.copyMakeBorder(img, border_size, border_size, border_size, border_size, cv2.BORDER_REPLICATE)
    result_img = np.zeros_like(img)
    for i in range(border_size, img_bordered.shape[0] - border_size):
        for j in range(border_size, img_bordered.shape[1] - border_size):

            window = img_bordered[i-border_size:i+border_size+1, j-border_size:j+border_size+1]
            
            equalized_window = cv2.equalizeHist(window)
            
            result_img[i-border_size, j-border_size] = equalized_window[border_size, border_size]

    return result_img
