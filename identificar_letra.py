import cv2
import numpy as np


# Celda seria la imagen de una letra
def identificar_letra(celda):
    """
    Función para identificar letras A, B, C y D basándose en el número de componentes conectadas.
    """
    # Convertir la celda a escala de grises
    celda_gris = cv2.cvtColor(celda, cv2.COLOR_BGR2GRAY)

    # Umbralizar la imagen para obtener una imagen binaria
    _, celda_binaria = cv2.threshold(celda_gris, 127, 255, cv2.THRESH_BINARY_INV)

    # Encontrar componentes conectadas
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(celda_binaria, 8, cv2.CV_32S)

    # Contar componentes
    componentes = num_labels - 1  # Restamos 1 para excluir el fondo

    # Identificar letra basada en el número de componentes
    if componentes == 3:
        # Podría ser A o B, se puede hacer una verificación adicional para diferenciarlas
        # Para simplificar, asumimos que si hay 3, es B
        letra_identificada = "B"
    elif componentes == 2:
        letra_identificada = "D"
    elif componentes == 1:
        letra_identificada = "C"
    else:
        letra_identificada = "A"  # Por defecto si no se identifica correctamente

    return letra_identificada