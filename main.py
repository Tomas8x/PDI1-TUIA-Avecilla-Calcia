from ecualizador_local import local_histogram_equalization
import cv2
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

if __name__ == "__main__":

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


