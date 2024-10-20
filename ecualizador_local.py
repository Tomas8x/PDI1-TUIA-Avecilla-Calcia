import cv2
import numpy as np
import matplotlib.pyplot as plt


def mostrar_imagenes(original, procesada):
    """
    Función para mostrar las imágenes original y procesada lado a lado.
    """
    plt.figure(figsize=(10, 5))
    
    # Imagen original
    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Imagen Original')
    plt.axis('off')
    
    # Imagen procesada
    plt.subplot(1, 2, 2)
    plt.imshow(procesada, cmap='gray')
    plt.title('Imagen Ecualizada Localmente')
    plt.axis('off')
    
    plt.show()

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


ruta_imagen = 'files/imagen_con_detalles_escondidos.tif'
window_sizes = [7, 14, 28, 52]

for window_size in window_sizes:

    imagen_procesada = local_histogram_equalization(ruta_imagen, window_size)

    # Cargar la imagen original para mostrarla
    imagen_original = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)

    # Mostrar las imágenes
    mostrar_imagenes(imagen_original, imagen_procesada)

    # Guardar la imagen procesada
    resultado_guardado = f'resultados/resultado_ecualizacion_window_{window_size}.png'
    cv2.imwrite(resultado_guardado, imagen_procesada)